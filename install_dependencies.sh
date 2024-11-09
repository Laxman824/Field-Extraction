#!/bin/bash

# Update package list
echo "Updating package list..."
apt-get update

# Install system dependencies
echo "Installing system dependencies..."
apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    python3-opencv

# Install Python packages
echo "Installing Python packages..."
pip uninstall -y opencv-python cv2 opencv-python-headless
pip install opencv-python-headless==4.7.0.72

echo "Dependencies installation complete!"
