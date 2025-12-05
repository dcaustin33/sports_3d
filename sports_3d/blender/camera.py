"""
Camera Setup for Blender Tennis Visualization

Creates and configures cameras including the main camera and
various preset viewing angles.
"""

import bpy
import math
from typing import List, Tuple

from .config import BlenderConfig


def create_camera(
    name: str,
    position: Tuple[float, float, float],
    look_at: Tuple[float, float, float],
    fov: float = 50.0
) -> bpy.types.Object:
    """Create a camera pointing at a target.

    Args:
        name: Camera object name
        position: Camera position (x, y, z) in meters
        look_at: Target point to look at (x, y, z)
        fov: Field of view in degrees

    Returns:
        Camera object
    """
    from mathutils import Vector

    bpy.ops.object.camera_add(location=position)
    camera = bpy.context.active_object
    camera.name = name

    # Point camera at target
    direction = Vector(look_at) - Vector(position)
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Set field of view
    camera.data.angle = math.radians(fov)

    return camera


def setup_main_camera(config: BlenderConfig) -> bpy.types.Object:
    """Create the main camera with user configuration.

    Args:
        config: Blender configuration

    Returns:
        Main camera object
    """
    camera = create_camera(
        name="MainCamera",
        position=config.camera_position,
        look_at=config.camera_look_at,
        fov=config.camera_fov
    )

    # Set as active scene camera
    bpy.context.scene.camera = camera

    return camera


def create_camera_presets() -> List[bpy.types.Object]:
    """Create multiple camera presets for different viewing angles.

    Creates cameras at various positions around the court for
    different perspectives.

    Returns:
        List of camera objects
    """
    presets = [
        # (name, position, look_at, fov)
        ("Cam_BroadcastNegZ", (0.0, 8.0, -22.0), (0, 1, 5), 40),
        ("Cam_BroadcastPosZ", (0.0, 8.0, 22.0), (0, 1, -5), 40),
        ("Cam_Courtside", (15.0, 3.0, 0.0), (0, 1, 0), 50),
        ("Cam_Overhead", (0.0, 20.0, 0.01), (0, 0, 0), 60),
        ("Cam_NetLevel", (8.0, 1.0, 0.0), (0, 1, 0), 70),
        ("Cam_BehindServer", (0.0, 2.5, 14.0), (0, 1, -5), 55),
    ]

    cameras = []
    for name, pos, look_at, fov in presets:
        camera = create_camera(name, pos, look_at, fov)
        cameras.append(camera)

    return cameras


def create_camera_parent(
    main_camera: bpy.types.Object,
    preset_cameras: List[bpy.types.Object]
) -> bpy.types.Object:
    """Create parent empty for camera organization.

    Args:
        main_camera: Main camera object
        preset_cameras: List of preset camera objects

    Returns:
        Parent empty object
    """
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0))
    cam_parent = bpy.context.active_object
    cam_parent.name = "Cameras"

    main_camera.parent = cam_parent
    for cam in preset_cameras:
        cam.parent = cam_parent

    return cam_parent


def set_active_camera(camera: bpy.types.Object) -> None:
    """Set a camera as the active scene camera.

    Args:
        camera: Camera object to make active
    """
    bpy.context.scene.camera = camera


def get_camera_by_name(name: str) -> bpy.types.Object:
    """Get a camera object by name.

    Args:
        name: Camera object name

    Returns:
        Camera object or None if not found
    """
    return bpy.data.objects.get(name)
