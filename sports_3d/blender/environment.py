"""
Environment Setup for Blender Tennis Visualization

Configures sky background, sun lighting, and fill lights for
a realistic outdoor tennis court environment.
"""

import bpy
import math
from typing import Optional

from .config import BlenderConfig


def setup_nishita_sky(config: BlenderConfig) -> None:
    """Set up procedural Nishita sky for outdoor environment.

    Uses Blender's built-in sky texture for realistic outdoor
    lighting and background.

    Args:
        config: Blender configuration
    """
    world = bpy.context.scene.world
    if world is None:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world

    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links

    nodes.clear()

    # Create nodes
    output = nodes.new('ShaderNodeOutputWorld')
    background = nodes.new('ShaderNodeBackground')
    sky_texture = nodes.new('ShaderNodeTexSky')

    # Configure Nishita sky
    sky_texture.sky_type = 'NISHITA'
    sky_texture.sun_elevation = math.radians(config.sky_sun_elevation)
    sky_texture.sun_rotation = math.radians(config.sky_sun_rotation)
    sky_texture.altitude = 0
    sky_texture.air_density = 1.0
    sky_texture.dust_density = 0.5
    sky_texture.ozone_density = 1.0

    # Set background strength
    background.inputs['Strength'].default_value = config.hdri_strength

    # Connect nodes
    links.new(sky_texture.outputs['Color'], background.inputs['Color'])
    links.new(background.outputs['Background'], output.inputs['Surface'])

    # Position nodes for readability
    output.location = (300, 0)
    background.location = (100, 0)
    sky_texture.location = (-200, 0)


def setup_sun_light(config: BlenderConfig) -> bpy.types.Object:
    """Create sun light for shadows and directional lighting.

    Args:
        config: Blender configuration

    Returns:
        Sun light object
    """
    # Create sun light matching sky sun position
    bpy.ops.object.light_add(type='SUN', location=(10, 20, 10))
    sun = bpy.context.active_object
    sun.name = "SunLight"

    sun.data.energy = config.sun_energy
    sun.data.angle = math.radians(0.545)  # Angular diameter of sun

    # Rotate to match sky sun direction
    sun.rotation_euler = (
        math.radians(90 - config.sky_sun_elevation),
        0,
        math.radians(config.sky_sun_rotation)
    )

    return sun


def setup_fill_light(config: BlenderConfig) -> bpy.types.Object:
    """Create fill light to soften shadows.

    Args:
        config: Blender configuration

    Returns:
        Fill light object
    """
    bpy.ops.object.light_add(type='AREA', location=(-10, 15, -5))
    fill = bpy.context.active_object
    fill.name = "FillLight"

    fill.data.energy = config.fill_light_energy
    fill.data.size = 10
    fill.data.shape = 'RECTANGLE'
    fill.data.size_y = 10

    # Point generally toward court center
    fill.rotation_euler = (math.radians(45), math.radians(-30), 0)

    return fill


def setup_environment(config: BlenderConfig) -> None:
    """Set up complete environment (sky + lights).

    Args:
        config: Blender configuration
    """
    # Create parent empty for lights
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    lights_parent = bpy.context.active_object
    lights_parent.name = "Lights"

    # Set up sky
    setup_nishita_sky(config)

    # Set up lights
    sun = setup_sun_light(config)
    sun.parent = lights_parent

    fill = setup_fill_light(config)
    fill.parent = lights_parent


def set_sky_sun_position(
    elevation: float,
    rotation: float,
    strength: float = 1.0
) -> None:
    """Update sky sun position dynamically.

    Args:
        elevation: Sun elevation in degrees
        rotation: Sun rotation in degrees
        strength: Background strength
    """
    world = bpy.context.scene.world
    if world is None or not world.use_nodes:
        return

    # Find sky texture node
    for node in world.node_tree.nodes:
        if node.type == 'TEX_SKY':
            node.sun_elevation = math.radians(elevation)
            node.sun_rotation = math.radians(rotation)
            break

    # Find background node
    for node in world.node_tree.nodes:
        if node.type == 'BACKGROUND':
            node.inputs['Strength'].default_value = strength
            break
