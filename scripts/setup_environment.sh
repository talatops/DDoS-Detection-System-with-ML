#!/bin/bash
# Setup script for DDoS Detection GPU Implementation
# Installs all required dependencies

set -e

echo "=== Installing system packages ==="
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libpcap-dev \
    tcpreplay \
    tshark \
    ocl-icd-opencl-dev \
    opencl-headers \
    clinfo \
    python3-pip \
    python3-dev \
    libboost-all-dev

echo "=== Installing NVIDIA OpenCL support ==="
# Check if NVIDIA drivers are installed
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers detected. Installing OpenCL support..."
    
    # Install NVIDIA OpenCL ICD package (this creates the proper ICD file)
    if ! dpkg -l | grep -q "^ii.*nvidia-opencl-icd"; then
        echo "Installing nvidia-opencl-icd package..."
        sudo apt-get install -y nvidia-opencl-icd || {
            echo "WARNING: Failed to install nvidia-opencl-icd"
            echo "You may need to install it manually: sudo apt-get install nvidia-opencl-icd"
        }
    else
        echo "NVIDIA OpenCL ICD already installed"
    fi
    
    # Update library cache
    sudo ldconfig
else
    echo "WARNING: nvidia-smi not found. NVIDIA drivers may not be installed."
    echo "Please install NVIDIA drivers first."
fi

echo "=== Installing Python packages ==="
# # Install system packages via apt (when available)
# sudo apt-get install -y \
#     python3-numpy \
#     python3-pandas \
#     python3-matplotlib \
#     python3-scipy \
#     python3-setuptools

# # Install remaining packages via pip3 (not available as apt packages or need newer versions)
# pip3 install --user \
#     scikit-learn \
#     flask \
#     flask-socketio \
#     psutil \
#     joblib \
#     scapy \
#     pybind11

echo "=== Verifying GPU detection ==="
if command -v clinfo &> /dev/null; then
    echo "Running clinfo..."
    clinfo | head -20
else
    echo "WARNING: clinfo not found. OpenCL may not be properly installed."
fi

if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,compute_cap --format=csv
else
    echo "WARNING: nvidia-smi not found. NVIDIA drivers may not be installed."
fi

echo "=== Setup complete ==="
echo "Next steps:"
echo "1. Verify OpenCL device with: clinfo"
echo "2. Test GPU access with a simple OpenCL program"
echo "3. Run: python3 scripts/prepare_data.py"

