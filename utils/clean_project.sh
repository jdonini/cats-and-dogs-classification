#!/bin/bash

find ~/Workspace/Stacking-Audio-Tagging/ -type d -name "*.pyc" -exec rm {} \;
find ~/Workspace/Stacking-Audio-Tagging/ -type d -name "*.DS_Store" -exec rm {} \;
find ~/Workspace/Stacking-Audio-Tagging/ -type d -name "*__pycache__" -exec rm -rf {} \;