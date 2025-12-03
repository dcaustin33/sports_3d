#!/bin/bash

python -m sports_3d.homography.kalman_inference \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_trajectory \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_events \
    --backup \
    --verbose;

# for f in *.bak; do mv "$f" "${f%.bak}"; done