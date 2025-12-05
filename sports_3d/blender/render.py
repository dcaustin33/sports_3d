"""
Render Settings for Blender Tennis Visualization

Configures EEVEE render engine settings for real-time preview
and final rendering.
"""

import bpy
from typing import Optional

from .config import BlenderConfig


def configure_eevee(config: BlenderConfig, num_frames: int) -> None:
    """Configure EEVEE render settings.

    Sets up EEVEE for good quality real-time preview and
    reasonable render times.

    Args:
        config: Blender configuration
        num_frames: Number of animation frames
    """
    scene = bpy.context.scene

    # Set render engine to EEVEE Next (Blender 4.x)
    scene.render.engine = 'BLENDER_EEVEE_NEXT'

    # Resolution
    scene.render.resolution_x = config.output_resolution[0]
    scene.render.resolution_y = config.output_resolution[1]
    scene.render.resolution_percentage = 100

    # Frame rate
    scene.render.fps = config.fps

    # Frame range
    scene.frame_start = 1
    scene.frame_end = num_frames
    scene.frame_current = 1

    # EEVEE quality settings
    scene.eevee.taa_render_samples = config.render_samples

    # Ambient occlusion
    scene.eevee.use_gtao = config.use_ambient_occlusion
    if config.use_ambient_occlusion:
        scene.eevee.gtao_distance = 0.5

    # Bloom
    scene.eevee.use_bloom = config.use_bloom
    if config.use_bloom:
        scene.eevee.bloom_intensity = config.bloom_intensity
        scene.eevee.bloom_threshold = 0.8


def set_output_path(
    output_path: str,
    file_format: str = 'FFMPEG',
    codec: str = 'H264'
) -> None:
    """Configure render output settings.

    Args:
        output_path: Path for output file/directory
        file_format: Output format ('FFMPEG', 'PNG', 'JPEG', etc.)
        codec: Video codec if using FFMPEG
    """
    scene = bpy.context.scene

    scene.render.filepath = output_path

    if file_format == 'FFMPEG':
        scene.render.image_settings.file_format = 'FFMPEG'
        scene.render.ffmpeg.format = 'MPEG4'
        scene.render.ffmpeg.codec = codec
        scene.render.ffmpeg.constant_rate_factor = 'HIGH'
        scene.render.ffmpeg.ffmpeg_preset = 'GOOD'
    elif file_format == 'PNG':
        scene.render.image_settings.file_format = 'PNG'
        scene.render.image_settings.color_mode = 'RGBA'
        scene.render.image_settings.color_depth = '8'
    elif file_format == 'JPEG':
        scene.render.image_settings.file_format = 'JPEG'
        scene.render.image_settings.quality = 90
    else:
        scene.render.image_settings.file_format = file_format


def set_viewport_shading(shading_type: str = 'MATERIAL') -> None:
    """Set viewport shading mode.

    Args:
        shading_type: 'WIREFRAME', 'SOLID', 'MATERIAL', or 'RENDERED'
    """
    for area in bpy.context.screen.areas:
        if area.type == 'VIEW_3D':
            for space in area.spaces:
                if space.type == 'VIEW_3D':
                    space.shading.type = shading_type


def render_frame(output_path: Optional[str] = None) -> None:
    """Render the current frame.

    Args:
        output_path: Optional output path override
    """
    if output_path:
        bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)


def render_animation(output_path: Optional[str] = None) -> None:
    """Render the full animation.

    Args:
        output_path: Optional output path override
    """
    if output_path:
        bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(animation=True)


def get_frame_info() -> dict:
    """Get current animation frame information.

    Returns:
        Dict with frame_start, frame_end, frame_current, fps
    """
    scene = bpy.context.scene
    return {
        'frame_start': scene.frame_start,
        'frame_end': scene.frame_end,
        'frame_current': scene.frame_current,
        'fps': scene.render.fps,
        'duration_seconds': (scene.frame_end - scene.frame_start + 1) / scene.render.fps,
    }
