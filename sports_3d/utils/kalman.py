"""
Trajectory smoothing for 3D sports tracking.

This module provides physics-aware filtering for noisy 3D trajectory measurements:
- Z-axis (depth): Exponential decay model anchored at event endpoints (monotonic)
- X-axis (lateral): Quadratic fit anchored at endpoints (allows spin curve, no weaving)
- Y-axis (vertical): Savitzky-Golay polynomial smoothing
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
    Event frames are included in BOTH the incoming and outgoing segments so that
    fits are anchored to the ground truth positions at discontinuities.

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
        if disc_idx >= start:
            segments.append((start, disc_idx + 1))
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

    Uses exponential decay model for Z-axis (monotonic), quadratic fit for X-axis
    (single curve, no weaving), and polynomial smoothing for Y-axis.
    Event positions anchor segment endpoints for X and Z.
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

    def _apply_y_smoothing(
        self,
        positions: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Apply polynomial smoothing to Y axis."""
        return self.poly_smoother.smooth(positions, segments)

    def _apply_x_quadratic_fit(
        self,
        timestamps: np.ndarray,
        x_positions: np.ndarray,
        uncertainties: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Apply quadratic fit to X-axis per-segment, anchored at endpoints.

        Model: x(t) = x_start + (x_end - x_start) * (t/T) + a * t * (T - t)
        - Linear interpolation between endpoints
        - Parabolic "bulge" term that's zero at both endpoints
        - Single parameter 'a' controls curvature (spin effect)

        This guarantees no weaving - a parabola can only curve one direction.

        Args:
            timestamps: (N,) array of timestamps in seconds
            x_positions: (N,) array of X positions in meters
            uncertainties: (N,) array of measurement uncertainties in meters
            segments: List of (start_idx, end_idx) for continuous segments

        Returns:
            x_filtered: (N,) filtered X positions
        """
        x_filtered = np.copy(x_positions)

        for start, end in segments:
            segment_length = end - start

            if segment_length < 2:
                continue

            if segment_length == 2:
                continue

            t_seg = timestamps[start:end]
            x_seg = x_positions[start:end]
            unc_seg = uncertainties[start:end]

            t_rel = t_seg - t_seg[0]
            T = t_rel[-1]

            x_start = x_seg[0]
            x_end = x_seg[-1]
            delta_x = x_end - x_start

            interior_mask = (t_rel > 0) & (t_rel < T)
            if not np.any(interior_mask):
                x_filtered[start:end] = x_start + delta_x * t_rel / T
                continue

            t_interior = t_rel[interior_mask]
            x_interior = x_seg[interior_mask]
            w_interior = 1.0 / (unc_seg[interior_mask] ** 2)

            x_linear_interior = x_start + delta_x * t_interior / T
            residuals = x_interior - x_linear_interior

            basis = t_interior * (T - t_interior)

            numerator = np.sum(w_interior * residuals * basis)
            denominator = np.sum(w_interior * basis ** 2)

            if abs(denominator) > 1e-10:
                a = numerator / denominator
            else:
                a = 0.0

            x_fit = x_start + delta_x * t_rel / T + a * t_rel * (T - t_rel)
            x_filtered[start:end] = x_fit

            if self.verbose:
                print(f"  Segment [{start}, {end}): X quadratic fit (curvature a={a:.4f})")

        return x_filtered

    def _apply_z_exponential_decay(
        self,
        timestamps: np.ndarray,
        z_positions: np.ndarray,
        uncertainties: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply exponential decay model to Z-axis per-segment.

        Physics model: Ball travels with air resistance causing velocity decay.
        z(t) = z0 + (v0/k) * (1 - e^(-k*t))
        v(t) = v0 * e^(-k*t)

        The model is anchored at segment endpoints (event positions are ground truth).
        This guarantees monotonic motion between events.

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

            if segment_length < 2:
                continue

            if segment_length == 2:
                dt = timestamps[start + 1] - timestamps[start]
                if dt > 0:
                    v = (z_positions[start + 1] - z_positions[start]) / dt
                    z_velocity[start] = v
                    z_velocity[start + 1] = v
                continue

            t_seg = timestamps[start:end]
            z_seg = z_positions[start:end]

            t_rel = t_seg - t_seg[0]
            T = t_rel[-1]

            z_start = z_seg[0]
            z_end = z_seg[-1]
            delta_z = z_end - z_start

            if abs(delta_z) < 1e-6:
                z_filtered[start:end] = z_start
                z_velocity[start:end] = 0.0
                continue

            k = self._fit_decay_constant(t_rel, z_seg, z_start, delta_z, T, uncertainties[start:end])

            if k > 1e-6:
                def z_model(t):
                    return z_start + delta_z * (1 - np.exp(-k * t)) / (1 - np.exp(-k * T))

                def v_model(t):
                    v0 = delta_z * k / (1 - np.exp(-k * T))
                    return v0 * np.exp(-k * t)

                z_fit = z_model(t_rel)
                v_fit = v_model(t_rel)
            else:
                v_const = delta_z / T
                z_fit = z_start + v_const * t_rel
                v_fit = np.full_like(t_rel, v_const)

            z_filtered[start:end] = z_fit
            z_velocity[start:end] = v_fit

            if self.verbose:
                v_start = v_fit[0]
                v_end = v_fit[-1]
                print(f"  Segment [{start}, {end}): Exponential decay fit "
                      f"(k={k:.3f}, v_start={v_start:.2f} m/s, v_end={v_end:.2f} m/s)")

        return z_filtered, z_velocity

    def _fit_decay_constant(
        self,
        t_rel: np.ndarray,
        z_seg: np.ndarray,
        z_start: float,
        delta_z: float,
        T: float,
        uncertainties: np.ndarray
    ) -> float:
        """
        Fit the decay constant k using weighted least squares on interior points.

        The model is: z(t) = z_start + delta_z * (1 - e^(-k*t)) / (1 - e^(-k*T))
        This is anchored at endpoints, so we fit k to minimize error on interior points.

        Args:
            t_rel: Relative timestamps (starting at 0)
            z_seg: Z positions for this segment
            z_start: Starting Z position (anchor)
            delta_z: Total Z displacement (z_end - z_start)
            T: Total segment duration
            uncertainties: Measurement uncertainties

        Returns:
            Optimal decay constant k (0 means use linear interpolation)
        """
        if len(t_rel) <= 2:
            return 0.0

        interior_mask = (t_rel > 0) & (t_rel < T)
        if not np.any(interior_mask):
            return 0.0

        t_interior = t_rel[interior_mask]
        z_interior = z_seg[interior_mask]
        w_interior = 1.0 / (uncertainties[interior_mask] ** 2)

        best_k = 0.0
        best_error = float('inf')

        for k in np.linspace(0.1, 5.0, 50):
            denom = 1 - np.exp(-k * T)
            if abs(denom) < 1e-10:
                continue

            z_pred = z_start + delta_z * (1 - np.exp(-k * t_interior)) / denom
            error = np.sum(w_interior * (z_pred - z_interior) ** 2)

            if error < best_error:
                best_error = error
                best_k = k

        z_linear = z_start + delta_z * t_interior / T
        linear_error = np.sum(w_interior * (z_linear - z_interior) ** 2)

        if linear_error <= best_error:
            return 0.0

        return best_k

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

        x_filtered = self._apply_x_quadratic_fit(
            timestamps,
            positions[:, 0],
            uncertainties,
            segments
        )
        y_filtered = self._apply_y_smoothing(positions[:, 1], segments)

        z_filtered, z_velocity = self._apply_z_exponential_decay(
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

