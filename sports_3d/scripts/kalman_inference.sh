#!/bin/bash

python -m sports_3d.homography.kalman_inference \
    /Users/derek/Desktop/sports_3d/data/sinner_ruud_trajectory \
    --backup \
    --verbose

# for f in *.bak; do mv "$f" "${f%.bak}"; done