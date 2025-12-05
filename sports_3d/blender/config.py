"""
Configuration for Blender Tennis Visualization

Contains all configurable parameters for the visualization including
court dimensions, colors, camera settings, and render options.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class BlenderConfig:
    """Configuration dataclass for Blender tennis visualization."""

    # Paths
    trajectory_dir: str = "data/sinner_ruud_trajectory"

    # Animation settings
    fps: int = 60
    ball_diameter_m: float = 0.066
    trail_length: int = 15
    trail_opacity_start: float = 0.1
    trail_opacity_end: float = 0.6
    trail_size_factor: float = 0.8  # Trail balls are smaller than main ball

    # Camera settings
    camera_position: Tuple[float, float, float] = (0.0, 3.0, -18.0)
    camera_look_at: Tuple[float, float, float] = (0.0, 1.0, 0.0)
    camera_fov: float = 50.0

    # Court colors (RGBA)
    court_inner_color: Tuple[float, float, float, float] = (0.05, 0.23, 0.40, 1.0)
    court_outer_color: Tuple[float, float, float, float] = (0.48, 0.72, 0.91, 1.0)
    line_color: Tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    ball_color: Tuple[float, float, float, float] = (0.9, 0.85, 0.0, 1.0)
    net_color: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)
    post_color: Tuple[float, float, float, float] = (0.3, 0.3, 0.3, 1.0)

    # Court dimensions (ITF official, in meters)
    court_length: float = 23.77
    court_doubles_width: float = 10.97
    court_singles_width: float = 8.23
    service_line_distance: float = 6.40
    net_height_center: float = 0.914
    net_height_posts: float = 1.07
    line_width: float = 0.05
    runoff_back: float = 6.4
    runoff_side: float = 3.66

    # Surface appearance
    court_roughness: float = 0.45
    court_noise_scale: float = 50.0
    court_noise_strength: float = 0.02

    # Environment settings
    sky_sun_elevation: float = 45.0
    sky_sun_rotation: float = 30.0
    hdri_strength: float = 1.0
    sun_energy: float = 3.0
    fill_light_energy: float = 500.0

    # Render settings (EEVEE)
    render_samples: int = 64
    output_resolution: Tuple[int, int] = (1920, 1080)
    use_bloom: bool = True
    bloom_intensity: float = 0.05
    use_ambient_occlusion: bool = True

    # Derived properties
    @property
    def half_court_length(self) -> float:
        return self.court_length / 2

    @property
    def half_doubles_width(self) -> float:
        return self.court_doubles_width / 2

    @property
    def half_singles_width(self) -> float:
        return self.court_singles_width / 2

    @property
    def ball_radius(self) -> float:
        return self.ball_diameter_m / 2


def get_default_config() -> BlenderConfig:
    """Return default configuration."""
    return BlenderConfig()
