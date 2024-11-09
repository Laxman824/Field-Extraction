#!/bin/bash

echo "Starting installation process..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    python3-pip \
    python3-venv \
    python3-dev \
    python3-distutils \
    python3-setuptools \
    tesseract-ocr \
    build-essential \
    libgl1-mesa-glx

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install torch first
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install core dependencies one by one
echo "Installing core dependencies..."
pip install numpy==1.23.5
pip install pillow==9.5.0
pip install opencv-python-headless==4.7.0.72
pip install streamlit==1.22.0

# Install DocTR
echo "Installing DocTR..."
pip install "python-doctr[torch]"

# Install other dependencies
echo "Installing additional dependencies..."
pip install pytesseract==0.3.10
pip install transformers
pip install datasets
pip install seqeval
pip install pyarrow==14.0.1
pip install python-dotenv
pip install pyyaml

echo "Installation complete!"
echo "Activate the virtual environment with: source venv/bin/activate"
