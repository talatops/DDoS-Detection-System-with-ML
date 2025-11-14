#!/bin/bash
# Script to install NVIDIA OpenCL support
# Run this if OpenCL shows 0 platforms after driver installation

set -e

echo "=== Installing NVIDIA CUDA Toolkit for OpenCL Support ==="
echo "This will install CUDA toolkit (~3GB download)"
echo "Press Ctrl+C to cancel, or Enter to continue..."
read

# Install CUDA toolkit
sudo apt-get update
sudo apt-get install -y nvidia-cuda-toolkit

# Create OpenCL vendors directory
sudo mkdir -p /etc/OpenCL/vendors

# Create NVIDIA ICD file
CUDA_OPENCL_LIB="/usr/local/cuda/lib64/libOpenCL.so"
if [ -f "$CUDA_OPENCL_LIB" ]; then
    echo "$CUDA_OPENCL_LIB" | sudo tee /etc/OpenCL/vendors/nvidia.icd > /dev/null
    echo "NVIDIA ICD file created: /etc/OpenCL/vendors/nvidia.icd"
else
    # Try alternative location
    CUDA_OPENCL_LIB=$(find /usr/local/cuda* -name "libOpenCL.so*" 2>/dev/null | head -1)
    if [ -n "$CUDA_OPENCL_LIB" ]; then
        echo "$CUDA_OPENCL_LIB" | sudo tee /etc/OpenCL/vendors/nvidia.icd > /dev/null
        echo "NVIDIA ICD file created: /etc/OpenCL/vendors/nvidia.icd"
    else
        echo "ERROR: Could not find CUDA OpenCL library"
        exit 1
    fi
fi

# Update library cache
sudo ldconfig

echo ""
echo "=== Verifying OpenCL installation ==="
clinfo | head -20

echo ""
echo "If you see 'Number of platforms: 1' or more, OpenCL is working!"

