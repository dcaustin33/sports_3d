"""
Tennis Ball and Animation for Blender

Creates tennis ball meshes and handles keyframe animation for
trajectory visualization, including ghost ball trail effect.
"""

import bpy
from typing import List, Tuple

from .config import BlenderConfig
from .materials import create_ball_material, create_transparent_material


def create_tennis_ball(
    config: BlenderConfig,
    name: str = "TennisBall"
) -> bpy.types.Object:
    """Create a tennis ball mesh.

    Args:
        config: Blender configuration
        name: Object name

    Returns:
        Tennis ball mesh object
    """
    radius = config.ball_radius

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=32,
        ring_count=16,
        location=(0, 0, 0)
    )
    ball = bpy.context.active_object
    ball.name = name

    bpy.ops.object.shade_smooth()

    mat = create_ball_material("TennisBallMaterial", config.ball_color)
    ball.data.materials.append(mat)

    return ball


def create_trail_ball(
    config: BlenderConfig,
    alpha: float,
    index: int
) -> bpy.types.Object:
    """Create a semi-transparent trail ball.

    Args:
        config: Blender configuration
        alpha: Opacity (0-1)
        index: Trail ball index (for naming)

    Returns:
        Trail ball mesh object
    """
    radius = config.ball_radius * config.trail_size_factor

    bpy.ops.mesh.primitive_uv_sphere_add(
        radius=radius,
        segments=16,
        ring_count=8,
        location=(0, -100, 0)  # Start off-screen
    )
    ball = bpy.context.active_object
    ball.name = f"TrailBall_{index:02d}"

    bpy.ops.object.shade_smooth()

    mat = create_transparent_material(
        f"TrailMaterial_{index:02d}",
        config.ball_color,
        alpha=alpha
    )
    ball.data.materials.append(mat)

    return ball


def create_trail_balls(
    config: BlenderConfig,
    count: int = None
) -> List[bpy.types.Object]:
    """Create a set of trail balls with graduated opacity.

    Args:
        config: Blender configuration
        count: Number of trail balls (default from config)

    Returns:
        List of trail ball objects, oldest first
    """
    if count is None:
        count = config.trail_length

    trail_balls = []

    for i in range(count):
        # Calculate opacity gradient (older = more transparent)
        t = i / max(1, count - 1)
        alpha = config.trail_opacity_start + t * (
            config.trail_opacity_end - config.trail_opacity_start
        )

        trail_ball = create_trail_ball(config, alpha, i)
        trail_balls.append(trail_ball)

    return trail_balls


def animate_object(
    obj: bpy.types.Object,
    positions: List[Tuple[float, float, float]],
    start_frame: int = 1
) -> None:
    """Animate an object along a series of positions.

    Args:
        obj: Blender object to animate
        positions: List of (x, y, z) position tuples
        start_frame: First frame number
    """
    for i, pos in enumerate(positions):
        frame = start_frame + i
        obj.location = (pos[0], pos[1], pos[2])
        obj.keyframe_insert(data_path="location", frame=frame)

    # Set linear interpolation for smooth motion
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for keyframe in fcurve.keyframe_points:
                keyframe.interpolation = 'LINEAR'


def animate_trail_ball(
    trail_ball: bpy.types.Object,
    positions: List[Tuple[float, float, float]],
    offset: int,
    start_frame: int = 1
) -> None:
    """Animate a trail ball with frame offset.

    The trail ball follows the main ball's path but is offset
    by a number of frames, creating the trailing effect.

    Args:
        trail_ball: Trail ball object
        positions: Full trajectory positions
        offset: Number of frames behind main ball
        start_frame: First frame number
    """
    # Create offset positions (pad with off-screen position)
    off_screen = (0, -100, 0)
    offset_positions = [off_screen] * offset + list(positions)

    # Trim to match original length
    offset_positions = offset_positions[:len(positions)]

    animate_object(trail_ball, offset_positions, start_frame)


def create_animated_ball_with_trail(
    config: BlenderConfig,
    positions: List[Tuple[float, float, float]],
    start_frame: int = 1
) -> Tuple[bpy.types.Object, List[bpy.types.Object]]:
    """Create main ball and trail balls with full animation.

    Args:
        config: Blender configuration
        positions: Trajectory positions list
        start_frame: First frame number

    Returns:
        Tuple of (main_ball, list_of_trail_balls)
    """
    # Create and animate main ball
    main_ball = create_tennis_ball(config)
    animate_object(main_ball, positions, start_frame)

    # Create and animate trail balls
    trail_balls = create_trail_balls(config)
    trail_length = len(trail_balls)

    for i, trail_ball in enumerate(trail_balls):
        # Older trail balls have larger offset (further behind)
        offset = trail_length - i
        animate_trail_ball(trail_ball, positions, offset, start_frame)

    return main_ball, trail_balls


def create_ball_parent(
    main_ball: bpy.types.Object,
    trail_balls: List[bpy.types.Object]
) -> bpy.types.Object:
    """Create parent empty for ball and trail organization.

    Args:
        main_ball: Main tennis ball object
        trail_balls: List of trail ball objects

    Returns:
        Parent empty object
    """
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    ball_parent = bpy.context.active_object
    ball_parent.name = "BallSystem"

    main_ball.parent = ball_parent
    for trail_ball in trail_balls:
        trail_ball.parent = ball_parent

    return ball_parent
