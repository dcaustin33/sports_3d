import csv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Optional


three_d_keypoints = [
    (-10.97 / 2, 0, 23.77 / 2),
    (10.97 / 2, 0, 23.77 / 2),
    (-10.97 / 2, 0, -23.77 / 2),
    (10.97 / 2, 0, -23.77 / 2),
    (-8.23 / 2, 0, 23.77 / 2),
    (-8.23 / 2, 0, -23.77 / 2),
    (8.23 / 2, 0, 23.77 / 2),
    (8.23 / 2, 0, -23.77 / 2),
    (-8.23 / 2, 0, 6.4),
    (8.23 / 2, 0, 6.4),
    (-8.23 / 2, 0, -6.4),
    (8.23 / 2, 0, -6.4),
    (0, 0, 6.4),
    (0, 0, -6.4),
    (0, 0.914, 0)
]


def save_trajectory_csv(trajectory_points: List[dict], output_path: str):
    """
    Save ball trajectory to CSV file.

    Args:
        trajectory_points: List of trajectory point dicts
        output_path: Output CSV file path
    """
    if not trajectory_points:
        print("Warning: No trajectory points to save")
        return

    fieldnames = ['frame_id', 'timestamp', 'x', 'y', 'z', 'depth', 'bbox_width_px', 'bbox_height_px']

    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for point in trajectory_points:
            row = {key: point.get(key, '') for key in fieldnames}
            writer.writerow(row)

    print(f"Saved {len(trajectory_points)} trajectory points to {output_path}")


def plot_trajectory_3d(
    trajectory_points: List[dict],
    calibration: dict,
    output_path: Optional[str] = None,
    show: bool = True
):
    """
    Create 3D visualization of ball trajectory with tennis court.

    Args:
        trajectory_points: List of trajectory dicts with x, y, z coordinates
        calibration: Calibration dict with camera_position
        output_path: Optional path to save plot as image
        show: Whether to display interactive plot window
    """
    if not trajectory_points:
        print("Warning: No trajectory points to plot")
        return

    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    x_coords = [p['x'] for p in trajectory_points]
    y_coords = [p['y'] for p in trajectory_points]
    z_coords = [p['z'] for p in trajectory_points]
    frame_ids = [p['frame_id'] for p in trajectory_points]

    scatter = ax.scatter(
        x_coords, y_coords, z_coords,
        c=frame_ids,
        cmap='viridis',
        s=50,
        alpha=0.8,
        edgecolors='black',
        linewidth=0.5
    )

    ax.plot(x_coords, y_coords, z_coords, 'r-', alpha=0.3, linewidth=1)

    ax.scatter(
        x_coords, [0] * len(x_coords), z_coords,
        c='gray',
        s=10,
        alpha=0.3,
        marker='o'
    )

    court_lines = [
        [0, 1, 3, 2, 0],
        [4, 6, 7, 5, 4],
        [8, 9, 11, 10, 8],
        [12, 13]
    ]

    for line_indices in court_lines:
        points = [three_d_keypoints[i] for i in line_indices]
        if points:
            xs, ys, zs = zip(*points)
            ax.plot(xs, ys, zs, 'g-', linewidth=2, alpha=0.6)

    if 'camera_position' in calibration:
        cam_pos = calibration['camera_position'].ravel()
        ax.scatter(
            [cam_pos[0]], [cam_pos[1]], [cam_pos[2]],
            c='red',
            s=200,
            marker='^',
            edgecolors='black',
            linewidth=2,
            label='Camera'
        )

    ax.set_xlabel('X (m) - Court Width', fontsize=12)
    ax.set_ylabel('Y (m) - Height', fontsize=12)
    ax.set_zlabel('Z (m) - Court Length', fontsize=12)
    ax.set_title('Ball Trajectory in 3D World Coordinates', fontsize=14, fontweight='bold')

    ax.set_xlim(-8, 8)
    ax.set_ylim(0, max(y_coords) + 2 if y_coords else 5)
    ax.set_zlim(-15, 15)

    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Frame ID', fontsize=11)

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def print_trajectory_statistics(trajectory_points: List[dict]):
    """
    Print summary statistics of trajectory.

    Args:
        trajectory_points: List of trajectory dicts
    """
    if not trajectory_points:
        print("No trajectory points to analyze")
        return

    x_coords = [p['x'] for p in trajectory_points]
    y_coords = [p['y'] for p in trajectory_points]
    z_coords = [p['z'] for p in trajectory_points]
    depths = [p['depth'] for p in trajectory_points]

    print("\n" + "="*60)
    print("TRAJECTORY STATISTICS")
    print("="*60)
    print(f"Total points: {len(trajectory_points)}")
    print(f"\nX (Court Width):")
    print(f"  Range: [{min(x_coords):.2f}, {max(x_coords):.2f}] m")
    print(f"  Mean: {np.mean(x_coords):.2f} m")
    print(f"\nY (Height):")
    print(f"  Range: [{min(y_coords):.2f}, {max(y_coords):.2f}] m")
    print(f"  Mean: {np.mean(y_coords):.2f} m")
    print(f"  Max height: {max(y_coords):.2f} m")
    print(f"\nZ (Court Length):")
    print(f"  Range: [{min(z_coords):.2f}, {max(z_coords):.2f}] m")
    print(f"  Mean: {np.mean(z_coords):.2f} m")
    print(f"\nDepth (Distance from camera):")
    print(f"  Range: [{min(depths):.2f}, {max(depths):.2f}] m")
    print(f"  Mean: {np.mean(depths):.2f} m")

    invalid_y = sum(1 for y in y_coords if y < 0)
    if invalid_y > 0:
        print(f"\n⚠ Warning: {invalid_y} points have Y < 0 (below ground)")

    far_x = sum(1 for x in x_coords if abs(x) > 15)
    if far_x > 0:
        print(f"⚠ Warning: {far_x} points have |X| > 15m (far from court)")

    print("="*60 + "\n")
