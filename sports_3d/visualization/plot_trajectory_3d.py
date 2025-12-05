"""
Interactive 3D trajectory visualization with frame-by-frame slider.

Usage:
    .venv/bin/python -m sports_3d.visualization.plot_trajectory_3d <trajectory_dir> [options]

Example:
    .venv/bin/python -m sports_3d.visualization.plot_trajectory_3d data/sinner_ruud_trajectory --tail_length 10
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import plotly.graph_objects as go

from sports_3d.assets_3d.tennis_court import TennisCourt3D
from sports_3d.utils.file_utils import discover_files_by_pattern, parse_frame_identifier


def normalize_camera_position(
    position_m: Tuple[float, float, float],
    court_bounds_m: Tuple[float, float, float] = (11.0, 5.0, 24.0),
) -> dict:
    """Convert camera position in meters to Plotly normalized coordinates.

    Plotly's camera 'eye' uses normalized coordinates relative to the scene bounds.
    This function converts a position in meters to the normalized system.

    Args:
        position_m: Camera position in meters (x, y, z)
        court_bounds_m: Scene bounds in meters (x_max, y_max, z_max) for normalization

    Returns:
        Dictionary with 'eye', 'center', and 'up' for Plotly camera
    """
    x_m, y_m, z_m = position_m
    x_bound, y_bound, z_bound = court_bounds_m

    # Normalize to [-1, 1] range based on court bounds
    eye_x = x_m / x_bound
    eye_y = y_m / y_bound
    eye_z = z_m / z_bound

    return {
        "eye": dict(x=eye_x, y=eye_y, z=eye_z),
        "center": dict(x=0, y=0, z=0),
        "up": dict(x=0, y=1, z=0),
    }


def load_trajectory_data(trajectory_dir: Path) -> Tuple[List[dict], np.ndarray]:
    """Load all trajectory JSONs and extract filtered positions.

    Args:
        trajectory_dir: Directory containing trajectory JSON files

    Returns:
        Tuple of (list of metadata dicts, positions array of shape (N, 3))
    """
    files = discover_files_by_pattern(trajectory_dir, "*_trajectory.json", "trajectory")

    metadata_list = []
    positions = []

    for file_path in files:
        with open(file_path) as f:
            data = json.load(f)

        parsed = parse_frame_identifier(file_path.name)
        frame_num = parsed[1] if parsed else 0
        timestamp = parsed[2] if parsed else 0.0

        filtered = data.get("filtered_projections", {})
        pos = filtered.get("position_filtered_m")

        if pos is None:
            continue

        positions.append([pos["x"], pos["y"], pos["z"]])
        metadata_list.append(
            {
                "frame_number": frame_num,
                "timestamp": timestamp,
                "file_path": str(file_path),
            }
        )

    return metadata_list, np.array(positions)




def create_trajectory_figure(
    positions: np.ndarray,
    metadata: List[dict],
    tail_length: int = 10,
    runoff_back: float = 6.4,
    camera_position_m: Optional[Tuple[float, float, float]] = None,
) -> go.Figure:
    """Create Plotly figure with animation frames for trajectory.

    Args:
        positions: Ball positions, shape (N, 3)
        metadata: List of frame metadata dicts
        tail_length: Number of trailing positions to show
        runoff_back: Distance behind baseline (default 6.4m, ITF minimum)
        camera_position_m: Optional camera position in meters (x, y, z)

    Returns:
        Plotly Figure with slider
    """
    court = TennisCourt3D(include_doubles=True)
    fig = court.to_plotly_figure(runoff_back=runoff_back)

    n_frames = len(positions)

    # Create animation frames
    frames = []
    slider_steps = []

    for i in range(n_frames):
        frame_data = []

        # Tail positions (fading opacity)
        tail_start = max(0, i - tail_length)
        tail_positions = positions[tail_start:i]

        if len(tail_positions) > 0:
            # Opacity gradient from 0.2 to 0.8
            opacities = np.linspace(0.2, 0.8, len(tail_positions))
            sizes = np.linspace(2, 4, len(tail_positions))

            frame_data.append(
                go.Scatter3d(
                    x=tail_positions[:, 0],
                    y=tail_positions[:, 1],
                    z=tail_positions[:, 2],
                    mode="markers+lines",
                    marker=dict(
                        size=sizes,
                        color="yellow",
                        opacity=0.6,
                    ),
                    line=dict(color="yellow", width=2),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

        # Current ball position
        frame_data.append(
            go.Scatter3d(
                x=[positions[i, 0]],
                y=[positions[i, 1]],
                z=[positions[i, 2]],
                mode="markers",
                marker=dict(size=6, color="yellow"),
                name=f"Frame {metadata[i]['frame_number']}",
                showlegend=False,
                hovertemplate=(
                    f"Frame: {metadata[i]['frame_number']}<br>"
                    f"Time: {metadata[i]['timestamp']:.3f}s<br>"
                    f"X: %{{x:.2f}}m<br>"
                    f"Y: %{{y:.2f}}m<br>"
                    f"Z: %{{z:.2f}}m"
                    "<extra></extra>"
                ),
            )
        )

        frames.append(go.Frame(data=frame_data, name=str(i)))

        slider_steps.append(
            {
                "args": [
                    [str(i)],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"},
                ],
                "label": str(metadata[i]["frame_number"]),
                "method": "animate",
            }
        )

    fig.frames = frames

    # Add initial frame data
    if n_frames > 0:
        for trace in frames[0].data:
            fig.add_trace(trace)

    # Set camera configuration
    if camera_position_m is not None:
        camera_config = normalize_camera_position(camera_position_m)
    else:
        # Default camera position (roughly -8.8m, 0.5m, -43.2m in world coords)
        camera_config = {
            "eye": dict(x=-0.8, y=0.1, z=-1.8),
            "center": dict(x=0, y=0, z=0),
            "up": dict(x=0, y=1, z=0),
        }

    # Add slider
    fig.update_layout(
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "currentvalue": {
                    "font": {"size": 16},
                    "prefix": "Frame: ",
                    "visible": True,
                    "xanchor": "right",
                },
                "steps": slider_steps,
                "x": 0.1,
                "len": 0.8,
                "y": 0,
                "pad": {"b": 10, "t": 50},
            }
        ],
        title=f"Ball Trajectory ({n_frames} frames)",
        scene=dict(camera=camera_config),
    )

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 3D trajectory visualization"
    )
    parser.add_argument(
        "trajectory_dir", type=Path, help="Directory with trajectory JSONs"
    )
    parser.add_argument(
        "--tail_length", type=int, default=10, help="Number of trailing positions"
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Output HTML file path"
    )
    parser.add_argument(
        "--show", action="store_true", help="Open in browser immediately"
    )
    parser.add_argument(
        "--flip_y", action="store_true", help="Flip Y axis (negate Y values)"
    )
    parser.add_argument(
        "--flip_x", action="store_true", help="Flip X axis (negate X values)"
    )
    parser.add_argument(
        "--y_scale",
        type=float,
        default=1.0,
        help="Scale factor for Y axis (e.g., 2.0 to double height)",
    )
    parser.add_argument(
        "--runoff_back",
        type=float,
        default=6.4,
        help="Distance behind baseline in meters (default 6.4m, ITF minimum)",
    )
    parser.add_argument(
        "--camera_x",
        type=float,
        default=None,
        help="Camera X position in meters",
    )
    parser.add_argument(
        "--camera_y",
        type=float,
        default=None,
        help="Camera Y position in meters",
    )
    parser.add_argument(
        "--camera_z",
        type=float,
        default=None,
        help="Camera Z position in meters",
    )

    args = parser.parse_args()

    print(f"Loading trajectories from {args.trajectory_dir}...")
    metadata, positions = load_trajectory_data(args.trajectory_dir)
    print(f"Loaded {len(positions)} frames")

    if args.flip_y:
        positions[:, 1] = -positions[:, 1]
        print("Flipped Y axis")

    if args.flip_x:
        positions[:, 0] = -positions[:, 0]
        print("Flipped X axis")

    if args.y_scale != 1.0:
        positions[:, 1] = positions[:, 1] * args.y_scale
        print(f"Scaled Y axis by {args.y_scale}x")

    # Set camera position if provided
    camera_position_m = None
    if all(v is not None for v in [args.camera_x, args.camera_y, args.camera_z]):
        camera_position_m = (args.camera_x, args.camera_y, args.camera_z)
        print(f"Camera position: ({args.camera_x}, {args.camera_y}, {args.camera_z}) meters")

    print("Creating visualization...")
    fig = create_trajectory_figure(
        positions, metadata, args.tail_length, args.runoff_back, camera_position_m
    )

    output_path = args.output or args.trajectory_dir / "trajectory_3d.html"
    fig.write_html(str(output_path))
    print(f"Saved to {output_path}")

    if args.show:
        fig.show()


if __name__ == "__main__":
    main()
