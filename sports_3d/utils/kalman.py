"""
Trajectory smoothing for 3D sports tracking.

This module provides physics-aware filtering for noisy 3D trajectory measurements:
- Z-axis (horizontal/depth): Quadratic fitting with velocity decay constraint
- X/Y-axes (horizontal): Savitzky-Golay polynomial smoothing
- Automatic detection of velocity discontinuities (bounces, hits)
- Adaptive measurement uncertainty based on reprojection error

Usage:
    from sports_3d.utils.kalman import TrajectoryFilter, reprojection_to_3d_uncertainty

    # Compute measurement uncertainties
    uncertainties = np.array([
        reprojection_to_3d_uncertainty(error_px, focal_px, distance_m)
        for error_px, focal_px, distance_m in zip(errors, focals, distances)
    ])

    # Filter trajectory
    filter = TrajectoryFilter(window_size_xy=7, poly_order=2)
    result = filter.filter(timestamps, positions, uncertainties)

    # Access filtered data
    filtered_positions = result['positions_filtered']
    velocities = result['velocities']
    discontinuity_frames = result['discontinuity_frames']
"""

from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import savgol_filter


def reprojection_to_3d_uncertainty(
    reprojection_error_px: float,
    focal_length_px: float,
    distance_m: float
) -> float:
    """
    Estimate 3D position uncertainty from 2D reprojection error.

    This approximates the angular error in camera calibration and projects it
    to 3D world space at the given distance.

    Args:
        reprojection_error_px: RMS reprojection error in pixels
        focal_length_px: Camera focal length in pixels
        distance_m: Distance from camera to object in meters

    Returns:
        Estimated 3D position uncertainty in meters

    Example:
        uncertainty_m = reprojection_to_3d_uncertainty(
            reprojection_error_px=93.6,
            focal_length_px=6724.8,
            distance_m=9.72
        )
    """
    angular_error = reprojection_error_px / focal_length_px
    return angular_error * distance_m


def _detect_discontinuities(
    n_frames: int,
    event_dict: Dict[int, Dict]
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Detect velocity discontinuities using event annotations.

    Uses event frames as ground truth discontinuities to split trajectory into segments.

    Args:
        n_frames: Total number of frames in trajectory
        event_dict: Dictionary mapping frame indices to event data

    Returns:
        discontinuity_indices: Array of frame indices where discontinuities occur
        segments: List of (start_idx, end_idx) tuples for continuous segments
    Segments with fewer than 3 points are excluded (insufficient for filtering).
    """
    if n_frames < 3:
        return np.array([], dtype=int), []

    discontinuity_indices = np.array(sorted(event_dict.keys()), dtype=int)

    segments = []
    start = 0
    for disc_idx in discontinuity_indices:
        if disc_idx > start:
            segments.append((start, disc_idx))
        start = disc_idx
    if start < n_frames:
        segments.append((start, n_frames))

    valid_segments = [(s, e) for s, e in segments if e - s >= 3]

    return discontinuity_indices, valid_segments


class PolynomialSmoother:
    """
    Savitzky-Golay polynomial smoothing for X/Y trajectory components.

    Applies scipy's savgol_filter per-segment to respect velocity discontinuities.
    """

    def __init__(self, window_size: int = 7, polynomial_order: int = 2):
        """
        Initialize polynomial smoother.

        Args:
            window_size: Smoothing window size (must be odd, >= poly_order + 1)
            polynomial_order: Polynomial degree (typically 2 or 3)
        """
        if window_size % 2 == 0:
            window_size += 1
        if window_size < polynomial_order + 1:
            window_size = polynomial_order + 1
            if window_size % 2 == 0:
                window_size += 1

        self.window_size = window_size
        self.polynomial_order = polynomial_order

    def smooth(
        self,
        positions: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Smooth position data per-segment.

        Points outside segments (gaps) are left unsmoothed.

        Args:
            positions: (N,) array of positions
            segments: List of (start_idx, end_idx) for continuous segments

        Returns:
            (N,) array of smoothed positions
        """
        smoothed = np.copy(positions)

        if len(segments) == 0:
            return smoothed

        for start, end in segments:
            segment_length = end - start
            if segment_length < 3:
                continue

            window = min(self.window_size, segment_length)
            if window % 2 == 0:
                window -= 1
            if window < 3:
                window = 3

            poly_order = min(self.polynomial_order, window - 1)

            segment_data = positions[start:end]
            smoothed_segment = savgol_filter(
                segment_data,
                window_length=window,
                polyorder=poly_order,
                mode='interp'
            )
            smoothed[start:end] = smoothed_segment

        return smoothed


class TrajectoryFilter:
    """
    Combined filter for 3D trajectory data.

    Uses quadratic fitting with velocity decay for Z-axis and polynomial smoothing for X/Y.
    Automatically detects and respects velocity discontinuities (bounces, hits).
    """

    def __init__(
        self,
        window_size_xy: int = 7,
        poly_order: int = 2,
        verbose: bool = False
    ):
        """
        Initialize trajectory filter.

        Args:
            window_size_xy: Smoothing window size for X/Y axes
            poly_order: Polynomial order for X/Y smoothing
            verbose: Enable debug output
        """
        self.verbose = verbose
        self.poly_smoother = PolynomialSmoother(window_size_xy, poly_order)

    def _apply_xy_smoothing(
        self,
        positions: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Apply polynomial smoothing to X or Y axis."""
        return self.poly_smoother.smooth(positions, segments)

    def _apply_z_quadratic_fit(
        self,
        timestamps: np.ndarray,
        z_positions: np.ndarray,
        uncertainties: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply quadratic fitting to Z-axis per-segment with velocity decay constraint.

        Physics model: Ball travels horizontally with air resistance, velocity decays.
        - Initial velocity is highest (from hit/bounce)
        - Velocity magnitude should decrease over time
        - Fit parabola z(t) = a*t^2 + b*t + c with weighted least squares

        Args:
            timestamps: (N,) array of timestamps in seconds
            z_positions: (N,) array of Z positions in meters
            uncertainties: (N,) array of measurement uncertainties in meters
            segments: List of (start_idx, end_idx) for continuous segments

        Returns:
            z_filtered: (N,) filtered Z positions
            z_velocity: (N,) estimated Z velocities
        """
        n = len(timestamps)
        z_filtered = np.copy(z_positions)
        z_velocity = np.zeros(n)

        for start, end in segments:
            segment_length = end - start

            if segment_length < 3:
                # Too short for quadratic fit, use raw positions
                if segment_length >= 2:
                    dt = timestamps[start + 1] - timestamps[start]
                    if dt > 0:
                        z_velocity[start] = (z_positions[start + 1] - z_positions[start]) / dt
                        z_velocity[start + 1] = z_velocity[start]
                continue

            # Extract segment data
            t_seg = timestamps[start:end]
            z_seg = z_positions[start:end]
            unc_seg = uncertainties[start:end]

            # Convert to relative time (starts at 0)
            t_rel = t_seg - t_seg[0]

            # Weighted least squares: minimize sum((z_measured - z_fit)^2 / uncertainty^2)
            # Fit z(t) = a*t^2 + b*t + c
            weights = 1.0 / (unc_seg ** 2)
            W = np.diag(weights)

            # Design matrix: [t^2, t, 1]
            A = np.column_stack([t_rel ** 2, t_rel, np.ones_like(t_rel)])

            # Solve weighted least squares: (A^T W A) params = A^T W z
            try:
                ATA = A.T @ W @ A
                ATb = A.T @ W @ z_seg
                params = np.linalg.solve(ATA, ATb)
                a, b, c = params

                # Compute fitted positions and velocities
                z_fit = a * t_rel ** 2 + b * t_rel + c
                v_fit = 2 * a * t_rel + b  # Derivative: v(t) = 2a*t + b

                # Check velocity decay constraint: |v(t_end)| should be <= |v(t_start)|
                v_start = b  # At t=0
                v_end = 2 * a * t_rel[-1] + b  # At t=t_end

                if abs(v_end) > abs(v_start) * 1.5:
                    # Velocity increasing too much - likely bad fit
                    # Fallback to linear fit: z(t) = b*t + c
                    A_linear = np.column_stack([t_rel, np.ones_like(t_rel)])
                    ATA_linear = A_linear.T @ W @ A_linear
                    ATb_linear = A_linear.T @ W @ z_seg
                    params_linear = np.linalg.solve(ATA_linear, ATb_linear)
                    b_lin, c_lin = params_linear

                    z_fit = b_lin * t_rel + c_lin
                    v_fit = np.full_like(t_rel, b_lin)  # Constant velocity

                    if self.verbose:
                        print(f"  Segment [{start}, {end}): Velocity constraint violated, "
                              f"using linear fit (v={b_lin:.2f} m/s)")
                elif self.verbose:
                    print(f"  Segment [{start}, {end}): Quadratic fit "
                          f"(v_start={v_start:.2f} m/s, v_end={v_end:.2f} m/s)")

                z_filtered[start:end] = z_fit
                z_velocity[start:end] = v_fit

            except np.linalg.LinAlgError:
                # Singular matrix - fallback to raw positions
                if self.verbose:
                    print(f"  Segment [{start}, {end}): Fit failed, using raw positions")
                # Velocity estimates from finite differences
                for i in range(start, end):
                    if i < end - 1:
                        dt = timestamps[i + 1] - timestamps[i]
                        if dt > 0:
                            z_velocity[i] = (z_positions[i + 1] - z_positions[i]) / dt

        return z_filtered, z_velocity

    def filter(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        uncertainties: np.ndarray,
        event_dict: Dict[int, Dict]
    ) -> Dict[str, np.ndarray]:
        """
        Filter 3D trajectory data.

        Args:
            timestamps: (N,) array of timestamps in seconds
            positions: (N, 3) array of [x, y, z] positions in meters
            uncertainties: (N,) array of measurement uncertainties in meters
            event_dict: Dictionary mapping frame indices to event data

        Returns:
            Dictionary with filtered results:
                - timestamps: (N,) original timestamps
                - positions_filtered: (N, 3) filtered positions
                - velocities: (N, 3) estimated velocities
                - x_raw, y_raw, z_raw: (N,) original positions per axis
                - x_filtered, y_filtered, z_filtered: (N,) filtered positions
                - z_velocity: (N,) Z-axis velocity from quadratic fitting
                - discontinuity_frames: Array of discontinuity indices
                - outlier_frames: Empty list (kept for backwards compatibility)
                - segments: List of (start, end) segment boundaries
        """
        n = len(timestamps)
        if n < 3:
            raise ValueError("Need at least 3 measurements for filtering")

        discontinuity_frames, segments = _detect_discontinuities(n, event_dict)

        if len(segments) == 0:
            if self.verbose:
                print("No segments detected, returning original positions")
            return {
                'timestamps': timestamps,
                'positions_filtered': positions.copy(),
                'velocities': np.zeros((n, 3)),
                'x_raw': positions[:, 0],
                'y_raw': positions[:, 1],
                'z_raw': positions[:, 2],
                'x_filtered': positions[:, 0],
                'y_filtered': positions[:, 1],
                'z_filtered': positions[:, 2],
                'z_velocity': np.zeros(n),
                'discontinuity_frames': discontinuity_frames,
                'outlier_frames': [],
                'segments': segments
            }

        if self.verbose:
            print(f"Detected {len(discontinuity_frames)} discontinuities at frames: {discontinuity_frames}")
            print(f"Trajectory split into {len(segments)} segments: {segments}")

        x_filtered = self._apply_xy_smoothing(positions[:, 0], segments)
        y_filtered = self._apply_xy_smoothing(positions[:, 1], segments)

        z_filtered, z_velocity = self._apply_z_quadratic_fit(
            timestamps,
            positions[:, 2],
            uncertainties,
            segments
        )

        positions_filtered = np.column_stack([x_filtered, y_filtered, z_filtered])

        velocities = np.zeros((n, 3))
        for i in range(1, n - 1):
            dt = timestamps[i + 1] - timestamps[i - 1]
            if dt > 0:
                velocities[i, 0] = (x_filtered[i + 1] - x_filtered[i - 1]) / dt
                velocities[i, 1] = (y_filtered[i + 1] - y_filtered[i - 1]) / dt
        velocities[:, 2] = z_velocity

        velocities[0] = velocities[1]
        velocities[-1] = velocities[-2]

        return {
            'timestamps': timestamps,
            'positions_filtered': positions_filtered,
            'velocities': velocities,
            'x_raw': positions[:, 0],
            'y_raw': positions[:, 1],
            'z_raw': positions[:, 2],
            'x_filtered': x_filtered,
            'y_filtered': y_filtered,
            'z_filtered': z_filtered,
            'z_velocity': z_velocity,
            'discontinuity_frames': discontinuity_frames,
            'outlier_frames': [],
            'segments': segments
        }

