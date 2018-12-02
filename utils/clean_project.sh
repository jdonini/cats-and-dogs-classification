#!/bin/bash

find ~/Workspace/Cats-and-Dogs-Classification/ -type d -name "*.pyc" -exec rm {} \;
find ~/Workspace/Cats-and-Dogs-Classification/ -type d -name "*.DS_Store" -exec rm {} \;
find ~/Workspace/Cats-and-Dogs-Classification/ -type d -name "*__pycache__" -exec rm -rf {} \;
