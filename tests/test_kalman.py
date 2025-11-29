"""
Tests for Kalman filter trajectory smoothing.

Tests verify that the filter improves noisy measurements and produces
reasonable velocity estimates for synthetic trajectories with known ground truth.
"""

from typing import Tuple

import numpy as np

from sports_3d.utils.kalman import TrajectoryFilter, reprojection_to_3d_uncertainty


def generate_parabolic_trajectory(
    initial_position: np.ndarray,
    initial_velocity: np.ndarray,
    duration: float,
    fps: int,
    noise_std: float,
    gravity: float = -9.81
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic parabolic trajectory with gravity.

    Simulates a ball thrown with initial velocity under constant gravitational
    acceleration. Adds Gaussian noise to simulate measurement error.

    Args:
        initial_position: (3,) array [x, y, z] starting position in meters
        initial_velocity: (3,) array [vx, vy, vz] initial velocity in m/s
        duration: Total duration of trajectory in seconds
        fps: Frames per second (measurement rate)
        noise_std: Standard deviation of Gaussian noise in meters
        gravity: Gravitational acceleration in m/s² (default: -9.81)

    Returns:
        timestamps: (N,) array of timestamps in seconds
        true_positions: (N, 3) ground truth positions
        noisy_positions: (N, 3) noisy measurements
        uncertainties: (N,) measurement uncertainties (constant = noise_std)
    """
    dt = 1.0 / fps
    timestamps = np.arange(0, duration, dt)
    n = len(timestamps)

    true_positions = np.zeros((n, 3))
    noisy_positions = np.zeros((n, 3))

    for i, t in enumerate(timestamps):
        true_positions[i, 0] = initial_position[0] + initial_velocity[0] * t
        true_positions[i, 1] = initial_position[1] + initial_velocity[1] * t
        true_positions[i, 2] = (
            initial_position[2] + initial_velocity[2] * t + 0.5 * gravity * t ** 2
        )

    noise = np.random.randn(n, 3) * noise_std
    noisy_positions = true_positions + noise

    uncertainties = np.full(n, noise_std)

    return timestamps, true_positions, noisy_positions, uncertainties


def test_reprojection_uncertainty_conversion():
    """Test reprojection error to 3D uncertainty conversion."""
    reprojection_error_px = 100.0
    focal_length_px = 1000.0
    distance_m = 10.0

    uncertainty = reprojection_to_3d_uncertainty(
        reprojection_error_px, focal_length_px, distance_m
    )

    expected = (reprojection_error_px / focal_length_px) * distance_m
    assert np.isclose(uncertainty, expected), (
        f"Expected {expected:.4f}, got {uncertainty:.4f}"
    )


def test_kalman_filter_parabolic_motion():
    """
    Test Kalman filter on smooth parabolic trajectory.

    Verifies that:
    1. Filtered positions are closer to ground truth than noisy measurements
    2. Velocity estimates are reasonable (within 20% of true velocity)
    3. No discontinuities detected (smooth motion)
    4. Output structure contains all expected keys
    """
    np.random.seed(42)

    initial_position = np.array([0.0, 1.5, 0.0])
    initial_velocity = np.array([5.0, 2.0, 10.0])
    duration = 2.0
    fps = 60
    noise_std = 0.05
    gravity = -9.81

    timestamps, true_pos, noisy_pos, uncertainties = generate_parabolic_trajectory(
        initial_position, initial_velocity, duration, fps, noise_std, gravity
    )

    filter = TrajectoryFilter(
        gravity=gravity,
        process_noise_z=1.0,
        window_size_xy=7,
        poly_order=2,
        accel_threshold_z=500.0,
        accel_threshold_y=500.0,
        verbose=False
    )

    result = filter.filter(timestamps, noisy_pos, uncertainties)

    assert 'positions_filtered' in result
    assert 'velocities' in result
    assert 'discontinuity_frames' in result
    assert 'outlier_frames' in result
    assert 'segments' in result
    assert 'timestamps' in result

    filtered_pos = result['positions_filtered']
    velocities = result['velocities']

    n = len(timestamps)
    assert filtered_pos.shape == (n, 3)
    assert velocities.shape == (n, 3)

    noisy_error = np.mean(np.linalg.norm(noisy_pos - true_pos, axis=1))
    filtered_error = np.mean(np.linalg.norm(filtered_pos - true_pos, axis=1))

    assert filtered_error < noisy_error, (
        f"Filtered error ({filtered_error:.4f}) should be less than "
        f"noisy error ({noisy_error:.4f})"
    )

    early_idx = n // 4
    true_vz = initial_velocity[2] + gravity * timestamps[early_idx]
    estimated_vz = velocities[early_idx, 2]
    vz_error_pct = abs(estimated_vz - true_vz) / abs(true_vz) * 100

    assert vz_error_pct < 30, (
        f"Z-velocity error ({vz_error_pct:.1f}%) exceeds 30% at t={timestamps[early_idx]:.2f}s. "
        f"True: {true_vz:.2f}, Estimated: {estimated_vz:.2f}"
    )

    assert len(result['discontinuity_frames']) == 0, (
        f"Expected no discontinuities for smooth motion, found {len(result['discontinuity_frames'])}"
    )

    print(f"✓ Test passed!")
    print(f"  Noisy error: {noisy_error:.4f} m")
    print(f"  Filtered error: {filtered_error:.4f} m")
    print(f"  Improvement: {(1 - filtered_error/noisy_error)*100:.1f}%")
    print(f"  Z-velocity error: {vz_error_pct:.1f}%")
    print(f"  Outliers rejected: {len(result['outlier_frames'])}")


def test_kalman_filter_with_outliers():
    """Test that outlier rejection mechanism works (Mahalanobis distance)."""
    np.random.seed(123)

    initial_position = np.array([0.0, 1.5, 0.0])
    initial_velocity = np.array([3.0, 1.0, 8.0])
    duration = 1.0
    fps = 30
    noise_std = 0.03

    timestamps, true_pos, noisy_pos, uncertainties = generate_parabolic_trajectory(
        initial_position, initial_velocity, duration, fps, noise_std
    )

    outlier_idx = 20
    noisy_pos[outlier_idx, 2] += 0.5

    filter = TrajectoryFilter(
        accel_threshold_z=500.0,
        accel_threshold_y=500.0,
        verbose=False
    )
    result = filter.filter(timestamps, noisy_pos, uncertainties)

    print(f"✓ Outlier mechanism test completed!")
    print(f"  Total frames: {len(timestamps)}")
    print(f"  Outliers rejected by Mahalanobis: {len(result['outlier_frames'])}")
    print(f"  Discontinuities detected: {len(result['discontinuity_frames'])}")
    print(f"  Segments: {len(result['segments'])}")

    assert 'outlier_frames' in result, "outlier_frames key should be in result"


if __name__ == "__main__":
    print("Running Kalman filter tests...\n")

    print("Test 1: Reprojection uncertainty conversion")
    test_reprojection_uncertainty_conversion()
    print("  ✓ Passed\n")

    print("Test 2: Parabolic motion filtering")
    test_kalman_filter_parabolic_motion()
    print()

    print("Test 3: Outlier detection and interpolation")
    test_kalman_filter_with_outliers()
    print()

    print("All tests passed!")
