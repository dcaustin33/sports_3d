#!/bin/bash

# Create trajectory visualization video with overlays
# Shows raw vs filtered 3D positions and velocities

python -m sports_3d.utils.tennis_video_util \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_frames \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_trajectory \
    /Users/derek/Desktop/sports_3d/output/trajectory_review.mp4 \
    --fps 1 \
    --overlay_position top_left \
    --verbose


