"""
Trajectory smoothing for 3D sports tracking.

This module provides physics-aware filtering for noisy 3D trajectory measurements:
- Z-axis (depth): Exponential decay model anchored at event endpoints (monotonic)
- X-axis (lateral): Quadratic fit anchored at endpoints (allows spin curve, no weaving)
- Y-axis (vertical): Ballistic model anchored at event endpoints (single parabola)
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


def _isotonic_increasing(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators algorithm for non-decreasing constraint.

    Finds the closest non-decreasing sequence to input by averaging violations.

    Args:
        y: Input array

    Returns:
        Non-decreasing array closest to y (minimizes squared error)
    """
    result = np.copy(y).astype(float)
    n = len(result)
    if n <= 1:
        return result

    i = 0
    while i < n - 1:
        if result[i] > result[i + 1]:
            j = i + 1
            while j < n and result[j] < result[j - 1]:
                j += 1
            block_start = i
            while block_start > 0 and result[block_start - 1] > result[j - 1]:
                block_start -= 1
            avg = np.mean(result[block_start:j])
            result[block_start:j] = avg
            i = block_start
        else:
            i += 1

    return result


def _isotonic_decreasing(y: np.ndarray) -> np.ndarray:
    """
    Pool Adjacent Violators algorithm for non-increasing constraint.

    Args:
        y: Input array

    Returns:
        Non-increasing array closest to y (minimizes squared error)
    """
    return -_isotonic_increasing(-y)


def _enforce_monotonic_trajectory(y_values: np.ndarray) -> np.ndarray:
    """
    Enforce single-apex trajectory using isotonic regression.

    For a ball trajectory, there should be at most one direction change:
    - Rising then falling (standard physics: Y increases upward)
    - OR falling then rising (inverted coords: Y increases downward, like this codebase)

    This function tries both patterns and picks the one with lower error.

    Args:
        y_values: Input Y positions

    Returns:
        Y positions with single-apex constraint enforced
    """
    n = len(y_values)
    if n < 3:
        return y_values.copy()

    def fit_apex_max(apex_idx: int) -> np.ndarray:
        """Fit increasing-then-decreasing (apex is maximum)."""
        y_result = np.copy(y_values)
        if apex_idx > 0:
            y_result[:apex_idx + 1] = _isotonic_increasing(y_values[:apex_idx + 1])
        if apex_idx < n - 1:
            y_result[apex_idx:] = _isotonic_decreasing(y_values[apex_idx:])
        return y_result

    def fit_apex_min(apex_idx: int) -> np.ndarray:
        """Fit decreasing-then-increasing (apex is minimum, for inverted Y)."""
        y_result = np.copy(y_values)
        if apex_idx > 0:
            y_result[:apex_idx + 1] = _isotonic_decreasing(y_values[:apex_idx + 1])
        if apex_idx < n - 1:
            y_result[apex_idx:] = _isotonic_increasing(y_values[apex_idx:])
        return y_result

    best_error = float('inf')
    best_result = y_values.copy()

    for apex_idx in range(n):
        for fit_func in [fit_apex_max, fit_apex_min]:
            y_fitted = fit_func(apex_idx)
            error = np.sum((y_fitted - y_values) ** 2)
            if error < best_error:
                best_error = error
                best_result = y_fitted

    return best_result


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
    (single curve, no weaving), and ballistic model for Y-axis.
    Event positions anchor segment endpoints for X, Y, and Z.
    """

    def __init__(
        self,
        window_size_xy: int = 7,
        poly_order: int = 2,
        y_fidelity: float = 0.5,
        verbose: bool = False
    ):
        """
        Initialize trajectory filter.

        Args:
            window_size_xy: Smoothing window size for X/Y axes
            poly_order: Polynomial order for X/Y smoothing
            y_fidelity: Blend factor for Y-axis (0.0 = pure ballistic, 1.0 = pure raw)
            verbose: Enable debug output
        """
        self.verbose = verbose
        self.y_fidelity = y_fidelity
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

    def _apply_y_ballistic(
        self,
        timestamps: np.ndarray,
        y_positions: np.ndarray,
        uncertainties: np.ndarray,
        segments: List[Tuple[int, int]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply ballistic model to Y-axis per-segment with two-pass filtering.

        Physics model: Ball vertical motion under effective gravity.
        y(t) = y_start + v_y0*t - (1/2)*g_eff*t²
        v(t) = v_y0 - g_eff*t

        Processing pipeline:
        1. Fit ballistic model (anchored at segment endpoints)
        2. Blend with raw measurements based on y_fidelity
        3. Enforce monotonic trajectory (single apex via isotonic regression)
        4. Apply Savitzky-Golay smoothing (remove high-frequency noise)
        5. Re-enforce monotonicity (guarantee no wavering from smoothing)

        Args:
            timestamps: (N,) array of timestamps in seconds
            y_positions: (N,) array of Y positions in meters
            uncertainties: (N,) array of measurement uncertainties in meters
            segments: List of (start_idx, end_idx) for continuous segments

        Returns:
            y_filtered: (N,) filtered Y positions
            y_velocity: (N,) estimated Y velocities
        """
        n = len(timestamps)
        y_filtered = np.copy(y_positions)
        y_velocity = np.zeros(n)
        
        flipped_y = False
        if np.mean(y_positions) < 0:
            # need to flip for gravity filtering
            y_positions = -y_positions
            flipped_y = True

        for start, end in segments:
            segment_length = end - start

            if segment_length < 2:
                continue

            if segment_length == 2:
                dt = timestamps[start + 1] - timestamps[start]
                if dt > 0:
                    v = (y_positions[start + 1] - y_positions[start]) / dt
                    y_velocity[start] = v
                    y_velocity[start + 1] = v
                continue

            t_seg = timestamps[start:end]
            y_seg = y_positions[start:end]

            t_rel = t_seg - t_seg[0]
            T = t_rel[-1]

            y_start_val = y_seg[0]
            y_end_val = y_seg[-1]
            delta_y = y_end_val - y_start_val

            if abs(delta_y) < 1e-6:
                y_filtered[start:end] = y_start_val
                y_velocity[start:end] = 0.0
                continue
            
            g_eff = self._fit_gravity_constant(
                t_rel, y_seg, y_start_val, delta_y, T, uncertainties[start:end]
            )

            if g_eff > 1e-6:
                v_y0 = delta_y / T + 0.5 * g_eff * T

                y_ballistic = y_start_val + v_y0 * t_rel - 0.5 * g_eff * (t_rel ** 2)
                v_fit = v_y0 - g_eff * t_rel
            else:
                v_const = delta_y / T
                y_ballistic = y_start_val + v_const * t_rel
                v_fit = np.full_like(t_rel, v_const)

            y_blended = (1 - self.y_fidelity) * y_ballistic + self.y_fidelity * y_seg

            y_monotonic = _enforce_monotonic_trajectory(y_blended)

            # Second pass: medium smoothing to remove high-frequency noise
            if segment_length >= 7:
                window = min(7, segment_length)
                if window % 2 == 0:
                    window -= 1

                y_smoothed = savgol_filter(
                    y_monotonic,
                    window_length=window,
                    polyorder=2,
                    mode='interp'
                )
                # Re-enforce monotonicity after smoothing
                y_final = _enforce_monotonic_trajectory(y_smoothed)
            else:
                # Segment too short for second smoothing pass
                y_final = y_monotonic

            y_filtered[start:end] = y_final
            y_velocity[start:end] = v_fit

            if self.verbose:
                v_start = v_fit[0]
                v_end = v_fit[-1]
                print(f"  Segment [{start}, {end}): Ballistic fit "
                      f"(g_eff={g_eff:.3f} m/s², v_start={v_start:.2f} m/s, "
                      f"v_end={v_end:.2f} m/s, fidelity={self.y_fidelity:.2f})")

        if flipped_y:
            y_filtered = -y_filtered
            y_velocity = -y_velocity

        return y_filtered, y_velocity

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

    def _fit_gravity_constant(
        self,
        t_rel: np.ndarray,
        y_seg: np.ndarray,
        y_start: float,
        delta_y: float,
        T: float,
        uncertainties: np.ndarray
    ) -> float:
        """
        Fit effective gravity constant g_eff using weighted least squares on interior points.

        The model is: y(t) = y_start + v_y0*t - (1/2)*g_eff*t²
        Where v_y0 = delta_y/T + (1/2)*g_eff*T (from endpoint constraint)

        This is anchored at endpoints, so we fit g_eff to minimize error on interior points.

        Args:
            t_rel: Relative timestamps (starting at 0)
            y_seg: Y positions for this segment
            y_start: Starting Y position (anchor)
            delta_y: Total Y displacement (y_end - y_start)
            T: Total segment duration
            uncertainties: Measurement uncertainties

        Returns:
            Optimal gravity constant g_eff in m/s² (0 means use linear interpolation)
        """
        if len(t_rel) <= 2:
            return 0.0

        interior_mask = (t_rel > 0) & (t_rel < T)
        if not np.any(interior_mask):
            return 0.0

        t_interior = t_rel[interior_mask]
        y_interior = y_seg[interior_mask]
        w_interior = 1.0 / (uncertainties[interior_mask] ** 2)

        best_g = 0.0
        best_error = float('inf')

        for g_eff in np.linspace(2.0, 25.0, 100):
            v_y0 = delta_y / T + 0.5 * g_eff * T

            y_pred = y_start + v_y0 * t_interior - 0.5 * g_eff * (t_interior ** 2)
            error = np.sum(w_interior * (y_pred - y_interior) ** 2)

            if error < best_error:
                best_error = error
                best_g = g_eff

        if best_g < 5.0:
            best_g = 5.0

        return best_g

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
                - y_velocity: (N,) Y-axis velocity from ballistic fitting
                - z_velocity: (N,) Z-axis velocity from exponential decay
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
                'y_velocity': np.zeros(n),
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
        y_filtered, y_velocity = self._apply_y_ballistic(
            timestamps,
            positions[:, 1],
            uncertainties,
            segments
        )

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
        velocities[:, 1] = y_velocity
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
            'y_velocity': y_velocity,
            'z_velocity': z_velocity,
            'discontinuity_frames': discontinuity_frames,
            'outlier_frames': [],
            'segments': segments
        }

