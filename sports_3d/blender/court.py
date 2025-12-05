"""
Tennis Court Geometry for Blender

Creates 3D tennis court geometry including playing surface, surround area,
court lines, and net. Uses ITF official dimensions.

Coordinate System:
    - Origin: Center of net
    - X-axis: Across court (sideline to sideline)
    - Y-axis: Height (vertical)
    - Z-axis: Along court (baseline to baseline)
"""

import bpy
import math
from typing import List, Tuple

from .config import BlenderConfig
from .materials import (
    create_textured_court_material,
    create_line_material,
    create_transparent_material,
    create_solid_material,
)


def create_court_surface(config: BlenderConfig) -> bpy.types.Object:
    """Create the inner court playing surface.

    Args:
        config: Blender configuration

    Returns:
        Court surface mesh object
    """
    half_length = config.half_court_length
    half_width = config.half_doubles_width

    bpy.ops.mesh.primitive_plane_add(size=1, location=(0, 0, 0))
    court = bpy.context.active_object
    court.name = "CourtSurface"
    court.scale = (half_width * 2, 1, half_length * 2)
    court.location = (0, -0.001, 0)  # Slightly below lines

    mat = create_textured_court_material(
        "CourtInner",
        config.court_inner_color,
        roughness=config.court_roughness,
        noise_scale=config.court_noise_scale,
        noise_strength=config.court_noise_strength,
    )
    court.data.materials.append(mat)

    bpy.ops.object.transform_apply(scale=True)

    return court


def create_surround_surface(config: BlenderConfig) -> List[bpy.types.Object]:
    """Create the surround/runoff area (4 strips around court).

    Args:
        config: Blender configuration

    Returns:
        List of surround strip mesh objects
    """
    half_length = config.half_court_length
    half_width = config.half_doubles_width
    runoff_back = config.runoff_back
    runoff_side = config.runoff_side

    mat = create_textured_court_material(
        "CourtOuter",
        config.court_outer_color,
        roughness=config.court_roughness,
        noise_scale=config.court_noise_scale,
        noise_strength=config.court_noise_strength,
    )

    strips = []

    # Back strip (positive Z - behind baseline)
    bpy.ops.mesh.primitive_plane_add(size=1)
    strip = bpy.context.active_object
    strip.name = "Surround_BackPos"
    strip.scale = ((half_width + runoff_side) * 2, 1, runoff_back)
    strip.location = (0, -0.002, half_length + runoff_back / 2)
    strip.data.materials.append(mat)
    bpy.ops.object.transform_apply(scale=True)
    strips.append(strip)

    # Back strip (negative Z - behind other baseline)
    bpy.ops.mesh.primitive_plane_add(size=1)
    strip = bpy.context.active_object
    strip.name = "Surround_BackNeg"
    strip.scale = ((half_width + runoff_side) * 2, 1, runoff_back)
    strip.location = (0, -0.002, -(half_length + runoff_back / 2))
    strip.data.materials.append(mat)
    bpy.ops.object.transform_apply(scale=True)
    strips.append(strip)

    # Side strip (positive X)
    bpy.ops.mesh.primitive_plane_add(size=1)
    strip = bpy.context.active_object
    strip.name = "Surround_SidePos"
    strip.scale = (runoff_side, 1, half_length * 2)
    strip.location = (half_width + runoff_side / 2, -0.002, 0)
    strip.data.materials.append(mat)
    bpy.ops.object.transform_apply(scale=True)
    strips.append(strip)

    # Side strip (negative X)
    bpy.ops.mesh.primitive_plane_add(size=1)
    strip = bpy.context.active_object
    strip.name = "Surround_SideNeg"
    strip.scale = (runoff_side, 1, half_length * 2)
    strip.location = (-(half_width + runoff_side / 2), -0.002, 0)
    strip.data.materials.append(mat)
    bpy.ops.object.transform_apply(scale=True)
    strips.append(strip)

    return strips


def create_court_line(
    start: Tuple[float, float, float],
    end: Tuple[float, float, float],
    width: float,
    name: str,
    material: bpy.types.Material
) -> bpy.types.Object:
    """Create a single court line as a thin box.

    Args:
        start: Start point (x, y, z)
        end: End point (x, y, z)
        width: Line width in meters
        name: Object name
        material: Line material

    Returns:
        Line mesh object
    """
    from mathutils import Vector

    start_vec = Vector(start)
    end_vec = Vector(end)

    center = (start_vec + end_vec) / 2
    direction = end_vec - start_vec
    length = direction.length

    bpy.ops.mesh.primitive_cube_add(size=1, location=(center.x, 0.001, center.z))
    line = bpy.context.active_object
    line.name = name

    # Scale based on line direction
    if abs(direction.x) > abs(direction.z):
        line.scale = (length, 0.002, width)
    else:
        line.scale = (width, 0.002, length)

    line.data.materials.append(material)
    bpy.ops.object.transform_apply(scale=True)

    return line


def create_court_lines(config: BlenderConfig) -> List[bpy.types.Object]:
    """Create all tennis court lines.

    Args:
        config: Blender configuration

    Returns:
        List of line mesh objects
    """
    half_length = config.half_court_length
    half_singles = config.half_singles_width
    half_doubles = config.half_doubles_width
    service_dist = config.service_line_distance
    line_width = config.line_width

    mat = create_line_material("LineWhite", config.line_color)
    lines = []

    # Baselines
    lines.append(create_court_line(
        (-half_doubles, 0, half_length), (half_doubles, 0, half_length),
        line_width, "Baseline_Pos", mat
    ))
    lines.append(create_court_line(
        (-half_doubles, 0, -half_length), (half_doubles, 0, -half_length),
        line_width, "Baseline_Neg", mat
    ))

    # Singles sidelines
    lines.append(create_court_line(
        (-half_singles, 0, -half_length), (-half_singles, 0, half_length),
        line_width, "SinglesSideline_Neg", mat
    ))
    lines.append(create_court_line(
        (half_singles, 0, -half_length), (half_singles, 0, half_length),
        line_width, "SinglesSideline_Pos", mat
    ))

    # Doubles sidelines
    lines.append(create_court_line(
        (-half_doubles, 0, -half_length), (-half_doubles, 0, half_length),
        line_width, "DoublesSideline_Neg", mat
    ))
    lines.append(create_court_line(
        (half_doubles, 0, -half_length), (half_doubles, 0, half_length),
        line_width, "DoublesSideline_Pos", mat
    ))

    # Service lines
    lines.append(create_court_line(
        (-half_singles, 0, service_dist), (half_singles, 0, service_dist),
        line_width, "ServiceLine_Pos", mat
    ))
    lines.append(create_court_line(
        (-half_singles, 0, -service_dist), (half_singles, 0, -service_dist),
        line_width, "ServiceLine_Neg", mat
    ))

    # Center service lines
    lines.append(create_court_line(
        (0, 0, 0), (0, 0, service_dist),
        line_width, "CenterServiceLine_Pos", mat
    ))
    lines.append(create_court_line(
        (0, 0, 0), (0, 0, -service_dist),
        line_width, "CenterServiceLine_Neg", mat
    ))

    # Center marks on baselines
    center_mark_len = 0.1
    lines.append(create_court_line(
        (0, 0, half_length), (0, 0, half_length - center_mark_len),
        line_width, "CenterMark_Pos", mat
    ))
    lines.append(create_court_line(
        (0, 0, -half_length), (0, 0, -half_length + center_mark_len),
        line_width, "CenterMark_Neg", mat
    ))

    return lines


def create_net(config: BlenderConfig) -> bpy.types.Object:
    """Create tennis net with posts.

    Args:
        config: Blender configuration

    Returns:
        Net parent object
    """
    half_width = config.half_doubles_width
    net_height = config.net_height_center
    post_height = config.net_height_posts

    # Create net mesh
    bpy.ops.mesh.primitive_cube_add(size=1, location=(0, net_height / 2, 0))
    net = bpy.context.active_object
    net.name = "Net"
    net.scale = (half_width * 2 + 0.5, net_height, 0.02)

    net_mat = create_transparent_material("NetMaterial", config.net_color, alpha=0.7)
    net.data.materials.append(net_mat)
    bpy.ops.object.transform_apply(scale=True)

    # Create posts
    post_mat = create_solid_material("PostMaterial", config.post_color, roughness=0.6)

    for side, x_pos in [("L", -(half_width + 0.3)), ("R", (half_width + 0.3))]:
        bpy.ops.mesh.primitive_cylinder_add(
            radius=0.04,
            depth=post_height,
            location=(x_pos, post_height / 2, 0)
        )
        post = bpy.context.active_object
        post.name = f"NetPost_{side}"
        post.data.materials.append(post_mat)
        post.parent = net

    return net


def create_tennis_court(config: BlenderConfig) -> bpy.types.Object:
    """Create complete tennis court with all components.

    Creates an empty parent object and parents all court components to it.

    Args:
        config: Blender configuration

    Returns:
        Court parent empty object
    """
    # Create parent empty
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    court_parent = bpy.context.active_object
    court_parent.name = "TennisCourt"

    # Create all court components
    court_surface = create_court_surface(config)
    court_surface.parent = court_parent

    surround_strips = create_surround_surface(config)
    for strip in surround_strips:
        strip.parent = court_parent

    court_lines = create_court_lines(config)
    for line in court_lines:
        line.parent = court_parent

    net = create_net(config)
    net.parent = court_parent

    return court_parent
