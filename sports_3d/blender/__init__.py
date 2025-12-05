"""
Blender Tennis Trajectory Visualization Package

This package provides tools for creating high-quality 3D visualizations
of tennis ball trajectories in Blender.

Usage:
    # From Blender's Python console or script:
    from sports_3d.blender import main
    main.run()

    # From command line:
    blender --python sports_3d/blender/main.py

    # With custom trajectory directory:
    blender --python sports_3d/blender/main.py -- --trajectory_dir /path/to/data
"""

from .config import BlenderConfig
from .main import run

__all__ = ['BlenderConfig', 'run']
