"""
Main Entry Point for Blender Tennis Visualization

This script orchestrates the creation of a complete tennis trajectory
visualization in Blender. Run this script from within Blender.

Usage (from Blender GUI):
    1. Open Blender 4.x
    2. Go to Scripting workspace
    3. Open this file
    4. Modify CONFIG below if needed
    5. Press Alt+P or click "Run Script"

Usage (from command line):
    blender --python sports_3d/blender/main.py

    With custom parameters:
    blender --python sports_3d/blender/main.py -- \\
        --trajectory_dir /path/to/data \\
        --camera_z -20.0
"""

import bpy
import sys
from pathlib import Path

# Add project root to path for imports when running from Blender
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from sports_3d.blender.config import BlenderConfig, get_default_config
from sports_3d.blender.data_loader import load_trajectory_data, get_position_bounds
from sports_3d.blender.court import create_tennis_court
from sports_3d.blender.ball import create_animated_ball_with_trail, create_ball_parent
from sports_3d.blender.camera import setup_main_camera, create_camera_presets, create_camera_parent
from sports_3d.blender.environment import setup_environment
from sports_3d.blender.render import configure_eevee, set_output_path


def clear_scene() -> None:
    """Remove all objects from the scene."""
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    # Clear orphan data blocks
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)
    for block in bpy.data.materials:
        if block.users == 0:
            bpy.data.materials.remove(block)
    for block in bpy.data.cameras:
        if block.users == 0:
            bpy.data.cameras.remove(block)
    for block in bpy.data.lights:
        if block.users == 0:
            bpy.data.lights.remove(block)


def print_header(text: str) -> None:
    """Print formatted header."""
    print(f"\n{'=' * 60}")
    print(text)
    print('=' * 60)


def print_step(step: int, total: int, text: str) -> None:
    """Print step progress."""
    print(f"\n[{step}/{total}] {text}")


def run(config: BlenderConfig = None) -> None:
    """Main function to create the visualization.

    Args:
        config: Blender configuration (uses default if None)
    """
    if config is None:
        config = get_default_config()

    print_header("BLENDER TENNIS TRAJECTORY VISUALIZATION")

    total_steps = 7

    # Step 1: Clear scene
    print_step(1, total_steps, "Clearing scene...")
    clear_scene()

    # Step 2: Load trajectory data
    print_step(2, total_steps, f"Loading trajectory from: {config.trajectory_dir}")
    metadata, positions = load_trajectory_data(config.trajectory_dir)

    if len(positions) == 0:
        print("ERROR: No trajectory data found!")
        print(f"       Check path: {config.trajectory_dir}")
        return

    print(f"       Loaded {len(positions)} positions")
    bounds = get_position_bounds(positions)
    print(f"       X range: {bounds['x_min']:.2f} to {bounds['x_max']:.2f} m")
    print(f"       Y range: {bounds['y_min']:.2f} to {bounds['y_max']:.2f} m")
    print(f"       Z range: {bounds['z_min']:.2f} to {bounds['z_max']:.2f} m")

    # Step 3: Create tennis court
    print_step(3, total_steps, "Creating tennis court...")
    court = create_tennis_court(config)

    # Step 4: Create animated ball with trail
    print_step(4, total_steps, "Creating animated ball with trail...")
    main_ball, trail_balls = create_animated_ball_with_trail(config, positions)
    ball_parent = create_ball_parent(main_ball, trail_balls)
    print(f"       Main ball + {len(trail_balls)} trail balls")

    # Step 5: Set up cameras
    print_step(5, total_steps, "Setting up cameras...")
    main_camera = setup_main_camera(config)
    preset_cameras = create_camera_presets()
    camera_parent = create_camera_parent(main_camera, preset_cameras)
    print(f"       Created {len(preset_cameras) + 1} cameras")

    # Step 6: Set up environment
    print_step(6, total_steps, "Setting up environment...")
    setup_environment(config)

    # Step 7: Configure render settings
    print_step(7, total_steps, "Configuring render settings...")
    configure_eevee(config, len(positions))

    # Print summary
    print_header("SETUP COMPLETE!")
    print(f"\nAnimation: {len(positions)} frames at {config.fps} fps")
    print(f"Duration: {len(positions) / config.fps:.2f} seconds")
    print(f"\nCamera position: {config.camera_position}")
    print(f"Looking at: {config.camera_look_at}")

    print("\n" + "-" * 60)
    print("AVAILABLE CAMERAS:")
    print("-" * 60)
    print("  - MainCamera (your configured view) [ACTIVE]")
    for cam in preset_cameras:
        print(f"  - {cam.name}")

    print("\n" + "-" * 60)
    print("KEYBOARD CONTROLS:")
    print("-" * 60)
    print("  Space        : Play/Pause animation")
    print("  Left/Right   : Previous/Next frame")
    print("  Shift+L/R    : Jump 10 frames")
    print("  Home         : Go to start")
    print("  End          : Go to end")
    print("  Numpad 0     : Camera view")
    print("  N            : Toggle sidebar (for camera switching)")
    print("  F12          : Render current frame")
    print("  Ctrl+F12     : Render animation")
    print("-" * 60)

    # Set viewport to camera view
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.region_3d.view_perspective = 'CAMERA'
                    space.shading.type = 'MATERIAL'
            break


def parse_args():
    """Parse command line arguments when run with -- separator."""
    if "--" not in sys.argv:
        return get_default_config()

    argv = sys.argv[sys.argv.index("--") + 1:]

    import argparse
    parser = argparse.ArgumentParser(description="Tennis Trajectory Visualization")
    parser.add_argument("--trajectory_dir", type=str, help="Path to trajectory JSONs")
    parser.add_argument("--fps", type=int, help="Frame rate")
    parser.add_argument("--camera_x", type=float, help="Camera X position")
    parser.add_argument("--camera_y", type=float, help="Camera Y position")
    parser.add_argument("--camera_z", type=float, help="Camera Z position")
    parser.add_argument("--trail_length", type=int, help="Number of trail balls")
    parser.add_argument("--fov", type=float, help="Camera field of view in degrees")

    args, _ = parser.parse_known_args(argv)

    config = get_default_config()

    if args.trajectory_dir:
        config.trajectory_dir = args.trajectory_dir
    if args.fps:
        config.fps = args.fps
    if args.trail_length:
        config.trail_length = args.trail_length
    if args.fov:
        config.camera_fov = args.fov

    # Update camera position if any coordinate is provided
    cam_pos = list(config.camera_position)
    if args.camera_x is not None:
        cam_pos[0] = args.camera_x
    if args.camera_y is not None:
        cam_pos[1] = args.camera_y
    if args.camera_z is not None:
        cam_pos[2] = args.camera_z
    config.camera_position = tuple(cam_pos)

    return config


# Run when script is executed
if __name__ == "__main__":
    config = parse_args()
    run(config)
