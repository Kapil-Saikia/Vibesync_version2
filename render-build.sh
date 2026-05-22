#!/usr/bin/env bash
# exit on error
set -o errexit

# Install Linux graphics libraries so MediaPipe won't crash
apt-get update && apt-get install -y libgl1-mesa-glx libgles2-mesa libglib2.0-0

# Install Python requirements
pip install -r requirements.txt