"""
Kalman filtering and trajectory smoothing for 3D sports tracking.

This module provides physics-aware filtering for noisy 3D trajectory measurements:
- Z-axis (vertical): Kalman filter with constant acceleration (gravity) model
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
    filter = TrajectoryFilter(gravity=-9.81, window_size_xy=7, poly_order=2)
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
    timestamps: np.ndarray,
    positions: np.ndarray,
    threshold_z: float = 200.0,
    threshold_y: float = 150.0
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Detect velocity discontinuities (bounces, hits) via acceleration spikes.

    Uses finite-difference approximation to compute velocities and accelerations
    in Y and Z directions. Detects frames where acceleration magnitude exceeds
    physical thresholds (much larger than gravity).

    Note: Thresholds should be set high enough to avoid false positives from
    measurement noise, which gets amplified by double differentiation.

    Args:
        timestamps: (N,) array of timestamps in seconds
        positions: (N, 3) array of [x, y, z] positions in meters
        threshold_z: Acceleration threshold for Z-axis in m/s² (default: 200)
        threshold_y: Acceleration threshold for Y-axis in m/s² (default: 150)

    Returns:
        discontinuity_indices: Array of frame indices where discontinuities occur
        segments: List of (start_idx, end_idx) tuples for continuous segments
    Segments with fewer than 3 points are excluded (insufficient for filtering).

    """
    n = len(timestamps)
    if n < 3:
        return np.array([], dtype=int), []
    if n < 5:
        return np.array([], dtype=int), [(0, n)]

    y_positions = positions[:, 1]
    z_positions = positions[:, 2]

    v_y = np.zeros(n)
    v_z = np.zeros(n)
    a_y = np.zeros(n)
    a_z = np.zeros(n)

    for i in range(1, n - 1):
        dt_total = timestamps[i + 1] - timestamps[i - 1]

        if dt_total > 0:
            v_y[i] = (y_positions[i + 1] - y_positions[i - 1]) / dt_total
            v_z[i] = (z_positions[i + 1] - z_positions[i - 1]) / dt_total

    for i in range(2, n - 2):
        dt_total = timestamps[i + 1] - timestamps[i - 1]
        if dt_total > 0:
            a_y[i] = (v_y[i + 1] - v_y[i - 1]) / dt_total
            a_z[i] = (v_z[i + 1] - v_z[i - 1]) / dt_total

    import pdb; pdb.set_trace()
    is_discontinuity = (np.abs(a_z) > threshold_z) | (np.abs(a_y) > threshold_y)
    discontinuity_indices = np.where(is_discontinuity)[0]

    segments = []
    if len(discontinuity_indices) == 0:
        segments.append((0, n))
    else:
        start = 0
        for disc_idx in discontinuity_indices:
            if disc_idx > start:
                segments.append((start, disc_idx))
            start = disc_idx + 1
        if start < n:
            segments.append((start, n))

    valid_segments = [(s, e) for s, e in segments if e - s >= 3]

    return discontinuity_indices, valid_segments


class KalmanFilter1D:
    """
    1D Kalman filter with constant acceleration model.

    State vector: [position, velocity, acceleration]
    Designed for vertical (Z-axis) motion with gravity as nominal acceleration.

    Attributes:
        state: (3,) array of [position, velocity, acceleration]
        covariance: (3, 3) state covariance matrix
        gravity: Acceleration due to gravity (m/s²)
    """

    def __init__(
        self,
        initial_position: float,
        initial_velocity: float = 0.0,
        gravity: float = -9.81,
        process_noise: float = 1.0,
        measurement_noise: float = 0.1,
        dt: float = 0.017
    ):
        """
        Initialize Kalman filter.

        Args:
            initial_position: Initial position estimate (m)
            initial_velocity: Initial velocity estimate (m/s)
            gravity: Acceleration due to gravity (m/s²)
            process_noise: Process noise standard deviation for acceleration
            measurement_noise: Initial measurement noise standard deviation (m)
            dt: Initial time step (s)
        """
        self.state = np.array([initial_position, initial_velocity, gravity])
        self.covariance = np.eye(3)
        self.covariance[0, 0] = measurement_noise ** 2
        self.covariance[1, 1] = 1.0
        self.covariance[2, 2] = process_noise ** 2
        self.gravity = gravity

        self.Q = np.diag([0.01, 0.5, process_noise ** 2])

    def predict(self, dt: float):
        """
        Predict next state using constant acceleration model.

        Args:
            dt: Time step since last update (s)
        """
        F = np.array([
            [1, dt, 0.5 * dt ** 2],
            [0, 1, dt],
            [0, 0, 1]
        ])

        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, measurement: float, measurement_noise: float) -> bool:
        """
        Update state estimate with new measurement using outlier rejection.

        Uses Mahalanobis distance (3-sigma test) to reject outliers. If rejected,
        state is not updated (prediction stands).

        Args:
            measurement: Measured position (m)
            measurement_noise: Measurement noise standard deviation (m)

        Returns:
            True if measurement was accepted, False if rejected as outlier
        """
        H = np.array([[1, 0, 0]])
        R = np.array([[measurement_noise ** 2]])

        y = measurement - H @ self.state
        S = H @ self.covariance @ H.T + R

        mahal_dist = np.abs(y[0]) / np.sqrt(S[0, 0])
        if mahal_dist > 3.0:
            return False

        K = self.covariance @ H.T @ np.linalg.inv(S)

        self.state = self.state + (K @ y).flatten()

        I = np.eye(3)
        I_KH = I - K @ H
        self.covariance = I_KH @ self.covariance @ I_KH.T + K @ R @ K.T

        return True

    def get_state(self) -> Tuple[float, float, float]:
        """
        Get current state estimate.

        Returns:
            (position, velocity, acceleration) in (m, m/s, m/s²)
        """
        return self.state[0], self.state[1], self.state[2]

    def reset(
        self,
        position: float,
        velocity: float = 0.0,
        measurement_noise: float = 0.1
    ):
        """
        Reset filter state (e.g., after discontinuity).

        Args:
            position: New position estimate (m)
            velocity: New velocity estimate (m/s)
            measurement_noise: Measurement uncertainty (m)
        """
        self.state = np.array([position, velocity, self.gravity])
        self.covariance = np.eye(3)
        self.covariance[0, 0] = measurement_noise ** 2
        self.covariance[1, 1] = 1.0
        self.covariance[2, 2] = self.Q[2, 2]


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

    Uses Kalman filter (with gravity) for Z-axis and polynomial smoothing for X/Y.
    Automatically detects and respects velocity discontinuities (bounces, hits).
    """

    def __init__(
        self,
        gravity: float = -9.81,
        process_noise_z: float = 1.0,
        window_size_xy: int = 7,
        poly_order: int = 2,
        accel_threshold_z: float = 200.0,
        accel_threshold_y: float = 150.0,
        verbose: bool = False
    ):
        """
        Initialize trajectory filter.

        Args:
            gravity: Acceleration due to gravity (m/s²)
            process_noise_z: Process noise for Z-axis Kalman filter
            window_size_xy: Smoothing window size for X/Y axes
            poly_order: Polynomial order for X/Y smoothing
            accel_threshold_z: Z-axis acceleration threshold for discontinuity detection (m/s²)
            accel_threshold_y: Y-axis acceleration threshold for discontinuity detection (m/s²)
            verbose: Enable debug output
        """
        self.gravity = gravity
        self.process_noise_z = process_noise_z
        self.accel_threshold_z = accel_threshold_z
        self.accel_threshold_y = accel_threshold_y
        self.verbose = verbose

        self.poly_smoother = PolynomialSmoother(window_size_xy, poly_order)

    def _apply_xy_smoothing(
        self,
        positions: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> np.ndarray:
        """Apply polynomial smoothing to X or Y axis."""
        return self.poly_smoother.smooth(positions, segments)

    def _apply_z_kalman(
        self,
        timestamps: np.ndarray,
        z_positions: np.ndarray,
        uncertainties: np.ndarray,
        segments: List[Tuple[int, int]]
    ) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Apply Kalman filter to Z-axis per-segment with outlier rejection.

        Outliers (Mahalanobis distance > 3) are rejected and linearly interpolated
        from neighboring valid measurements.

        Returns:
            z_filtered: (N,) filtered Z positions
            z_velocity: (N,) estimated Z velocities
            rejected_indices: List of frame indices rejected as outliers
        """
        n = len(timestamps)
        z_filtered = np.zeros(n)
        z_velocity = np.zeros(n)
        rejected_indices = []

        for start, end in segments:
            if end - start < 2:
                z_filtered[start:end] = z_positions[start:end]
                continue

            initial_pos = z_positions[start]
            initial_vel = 0.0
            if end - start >= 2:
                dt0 = timestamps[start + 1] - timestamps[start]
                if dt0 > 0:
                    initial_vel = (z_positions[start + 1] - z_positions[start]) / dt0

            kf = KalmanFilter1D(
                initial_position=initial_pos,
                initial_velocity=initial_vel,
                gravity=self.gravity,
                process_noise=self.process_noise_z,
                measurement_noise=uncertainties[start]
            )

            segment_rejected = []

            for i in range(start, end):
                if i > start:
                    dt = timestamps[i] - timestamps[i - 1]
                    kf.predict(dt)

                accepted = kf.update(z_positions[i], uncertainties[i])
                pos, vel, _ = kf.get_state()
                z_filtered[i] = pos
                z_velocity[i] = vel

                if not accepted:
                    segment_rejected.append(i)

            rejected_indices.extend(segment_rejected)

            if segment_rejected:
                for idx in segment_rejected:
                    valid_indices = [i for i in range(start, end) if i not in segment_rejected]
                    if len(valid_indices) >= 2:
                        z_filtered[idx] = np.interp(
                            timestamps[idx],
                            timestamps[valid_indices],
                            z_filtered[valid_indices]
                        )
                        z_velocity[idx] = np.interp(
                            timestamps[idx],
                            timestamps[valid_indices],
                            z_velocity[valid_indices]
                        )

        return z_filtered, z_velocity, rejected_indices

    def filter(
        self,
        timestamps: np.ndarray,
        positions: np.ndarray,
        uncertainties: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Filter 3D trajectory data.

        Args:
            timestamps: (N,) array of timestamps in seconds
            positions: (N, 3) array of [x, y, z] positions in meters
            uncertainties: (N,) array of measurement uncertainties in meters

        Returns:
            Dictionary with filtered results:
                - timestamps: (N,) original timestamps
                - positions_filtered: (N, 3) filtered positions
                - velocities: (N, 3) estimated velocities
                - x_raw, y_raw, z_raw: (N,) original positions per axis
                - x_filtered, y_filtered, z_filtered: (N,) filtered positions
                - z_velocity: (N,) vertical velocity from Kalman filter
                - discontinuity_frames: Array of discontinuity indices
                - outlier_frames: List of frame indices rejected as outliers
                - segments: List of (start, end) segment boundaries
        """
        n = len(timestamps)
        if n < 3:
            raise ValueError("Need at least 3 measurements for filtering")

        discontinuity_frames, segments = _detect_discontinuities(
            timestamps,
            positions,
            self.accel_threshold_z,
            self.accel_threshold_y
        )

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

        z_filtered, z_velocity, outlier_frames = self._apply_z_kalman(
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
            'outlier_frames': outlier_frames,
            'segments': segments
        }

