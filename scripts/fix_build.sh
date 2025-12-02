#!/bin/bash
# Fix build directory permissions and rebuild

set -e

cd "$(dirname "$0")/.."

echo "Fixing build directory permissions..."

# Remove old build directory
if [ -d "build" ]; then
    echo "Removing old build directory..."
    rm -rf build
fi

# Create new build directory with correct permissions
echo "Creating new build directory..."
mkdir -p build
chmod 755 build

# Configure with CMake
echo "Running CMake..."
cd build
cmake .. || {
    echo "ERROR: CMake failed. Trying with full path..."
    /usr/bin/cmake .. || {
        echo "ERROR: CMake still failed. Please check CMake installation."
        exit 1
    }
}

# Build
echo "Building project..."
make -j$(nproc) || {
    echo "ERROR: Build failed. Check errors above."
    exit 1
}

echo ""
echo "✓ Build successful!"
echo "  Executable: build/detector"
echo ""
echo "Test it with:"
echo "  ./build/detector --use-gpu --pcap /dev/null"

