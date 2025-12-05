#!/bin/bash

python -m sports_3d.homography.kalman_inference \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_trajectory \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_events \
    --backup \
    --verbose;

python -m sports_3d.visualization.plot_trajectory_3d \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_trajectory \
    --tail_length 5 \
    --flip_y \
    --flip_x \
    --y_scale 1.0 \
    --output /Users/derek/Desktop/sports_3d/data/html_output/sinner_ruud_trajectory.html \
    --runoff_back 10.0 \
    --camera_x 0.0 \
    --camera_y 0.0 \
    --camera_z -15.0;

# for f in *.bak; do mv "$f" "${f%.bak}"; done