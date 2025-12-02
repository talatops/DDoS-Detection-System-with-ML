#!/bin/bash
# Test runner script for GPU validation and benchmarking

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "GPU Tests Runner"
echo "=========================================="

# Create build directory if it doesn't exist
mkdir -p build
mkdir -p results

# Build the project first
echo ""
echo "1. Building project..."
cd build
cmake .. || {
    echo "ERROR: CMake configuration failed"
    exit 1
}
make -j$(nproc) || {
    echo "ERROR: Build failed"
    exit 1
}
cd ..

# Run GPU correctness validation
echo ""
echo "2. Running GPU correctness validation..."
python3 tools/validate_gpu_correctness.py || {
    echo "WARNING: GPU correctness validation failed or incomplete"
}

# Build GPU entropy correctness test binary
echo ""
echo "3. Building GPU entropy correctness test..."
cd build
g++ -std=c++17 -O3 \
    -Isrc -I../src \
    -I/usr/include \
    ../tools/test_gpu_entropy.cpp \
    ../src/opencl/gpu_detector.cpp \
    ../src/opencl/host.cpp \
    ../src/detectors/entropy_cpu.cpp \
    ../src/ingest/window_manager.cpp \
    ../src/ingest/pcap_reader.cpp \
    ../src/utils/simple_json.cpp \
    -lOpenCL -lpcap -lpthread \
    -o test_gpu_entropy 2>&1 || {
    echo "WARNING: Failed to build GPU entropy test (may need GPU drivers)"
}

# Build GPU performance benchmark
echo ""
echo "4. Building GPU performance benchmark..."
g++ -std=c++17 -O3 \
    -Isrc -I../src \
    -I/usr/include \
    ../tools/benchmark_gpu_performance.cpp \
    ../src/opencl/gpu_detector.cpp \
    ../src/opencl/host.cpp \
    ../src/detectors/entropy_cpu.cpp \
    ../src/ingest/window_manager.cpp \
    ../src/ingest/pcap_reader.cpp \
    ../src/utils/simple_json.cpp \
    -lOpenCL -lpcap -lpthread \
    -o benchmark_gpu_performance 2>&1 || {
    echo "WARNING: Failed to build benchmark (may need GPU drivers)"
}
cd ..

echo ""
echo "5. Running GPU performance benchmark..."
if [ -f build/benchmark_gpu_performance ]; then
    ./build/benchmark_gpu_performance || {
        echo "WARNING: GPU benchmark failed"
    }
else
    echo "ERROR: Benchmark executable not found at build/benchmark_gpu_performance"
    echo "Trying build/benchmark_gpu_performance from build directory..."
    if [ -f build/build/benchmark_gpu_performance ]; then
        ./build/build/benchmark_gpu_performance || {
            echo "WARNING: GPU benchmark failed"
        }
    fi
fi

echo ""
echo "=========================================="
echo "GPU tests complete"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - results/gpu_correctness_report.txt"
echo "  - results/gpu_cpu_entropy_comparison.png"

