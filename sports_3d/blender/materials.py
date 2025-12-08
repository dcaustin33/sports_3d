"""
Blender Material Utilities

Functions for creating various types of materials for the tennis
court visualization.
"""

import math
from typing import Tuple

import bpy


def create_solid_material(
    name: str,
    color: Tuple[float, float, float, float],
    roughness: float = 0.5,
    emission_strength: float = 0.0
) -> bpy.types.Material:
    """Create a solid color material with Principled BSDF.

    Args:
        name: Material name
        color: RGBA color tuple (0-1 range)
        roughness: Surface roughness (0=glossy, 1=matte)
        emission_strength: Optional emission for glowing effect

    Returns:
        Blender Material object
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')

    principled.inputs['Base Color'].default_value = color
    principled.inputs['Roughness'].default_value = roughness

    if emission_strength > 0:
        principled.inputs['Emission Color'].default_value = color
        principled.inputs['Emission Strength'].default_value = emission_strength

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    output.location = (300, 0)
    principled.location = (0, 0)

    return mat


def create_transparent_material(
    name: str,
    color: Tuple[float, float, float, float],
    alpha: float,
    roughness: float = 0.3
) -> bpy.types.Material:
    """Create a semi-transparent material.

    Args:
        name: Material name
        color: RGBA color tuple
        alpha: Transparency (0=invisible, 1=opaque)
        roughness: Surface roughness

    Returns:
        Blender Material object
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    mat.blend_method = 'BLEND'
    mat.shadow_method = 'HASHED'

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')

    principled.inputs['Base Color'].default_value = color
    principled.inputs['Alpha'].default_value = alpha
    principled.inputs['Roughness'].default_value = roughness

    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    output.location = (300, 0)
    principled.location = (0, 0)

    return mat


def create_textured_court_material(
    name: str,
    base_color: Tuple[float, float, float, float],
    roughness: float = 0.45,
    noise_scale: float = 50.0,
    noise_strength: float = 0.02
) -> bpy.types.Material:
    """Create a court material with subtle procedural texture.

    Adds a slight noise variation to the color for realism without
    being distracting.

    Args:
        name: Material name
        base_color: Base RGBA color
        roughness: Surface roughness
        noise_scale: Scale of noise texture (higher = finer grain)
        noise_strength: How much noise affects the color

    Returns:
        Blender Material object
    """
    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    nodes.clear()

    # Create nodes
    output = nodes.new('ShaderNodeOutputMaterial')
    principled = nodes.new('ShaderNodeBsdfPrincipled')
    mix_color = nodes.new('ShaderNodeMix')
    noise = nodes.new('ShaderNodeTexNoise')
    tex_coord = nodes.new('ShaderNodeTexCoord')
    color_ramp = nodes.new('ShaderNodeValToRGB')

    # Configure mix node
    mix_color.data_type = 'RGBA'
    mix_color.blend_type = 'MIX'
    mix_color.inputs['Factor'].default_value = noise_strength

    # Configure noise
    noise.inputs['Scale'].default_value = noise_scale
    noise.inputs['Detail'].default_value = 2.0
    noise.inputs['Roughness'].default_value = 0.5

    # Configure color ramp for subtle variation
    color_ramp.color_ramp.elements[0].color = (
        base_color[0] * 0.95,
        base_color[1] * 0.95,
        base_color[2] * 0.95,
        1.0
    )
    color_ramp.color_ramp.elements[1].color = (
        min(base_color[0] * 1.05, 1.0),
        min(base_color[1] * 1.05, 1.0),
        min(base_color[2] * 1.05, 1.0),
        1.0
    )

    # Set base color
    mix_color.inputs['A'].default_value = base_color

    # Set principled shader
    principled.inputs['Roughness'].default_value = roughness

    # Connect nodes
    links.new(tex_coord.outputs['Generated'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], mix_color.inputs['B'])
    links.new(mix_color.outputs['Result'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output.inputs['Surface'])

    # Position nodes
    tex_coord.location = (-800, 0)
    noise.location = (-600, 0)
    color_ramp.location = (-400, 0)
    mix_color.location = (-200, 0)
    principled.location = (0, 0)
    output.location = (300, 0)

    return mat


def create_line_material(
    name: str = "CourtLine",
    color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
) -> bpy.types.Material:
    """Create a white court line material.

    Args:
        name: Material name
        color: Line color (default white)

    Returns:
        Blender Material object
    """
    return create_solid_material(name, color, roughness=0.3)


def create_ball_material(
    name: str = "TennisBall",
    color: Tuple[float, float, float, float] = (0.9, 0.85, 0.0, 1.0)
) -> bpy.types.Material:
    """Create a tennis ball material with slight glow.

    Args:
        name: Material name
        color: Ball color (default tennis yellow)

    Returns:
        Blender Material object
    """
    return create_solid_material(name, color, roughness=0.4, emission_strength=0.1)
