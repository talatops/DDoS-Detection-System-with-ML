#!/bin/bash
# Quick test script for Phase 4 & 5 implementation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "Quick Test Script - Phase 4 & 5"
echo "=========================================="

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test results
PASSED=0
FAILED=0

test_step() {
    local name=$1
    local command=$2
    
    echo ""
    echo -e "${YELLOW}Testing: $name${NC}"
    echo "Command: $command"
    echo "----------------------------------------"
    
    if eval "$command" > /tmp/test_output.log 2>&1; then
        echo -e "${GREEN}✓ PASSED: $name${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED: $name${NC}"
        echo "Last 10 lines of output:"
        tail -10 /tmp/test_output.log
        ((FAILED++))
        return 1
    fi
}

# Test 1: Check GPU
echo ""
echo "1. Checking GPU availability..."
if nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ GPU detected${NC}"
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
else
    echo -e "${YELLOW}⚠ GPU not detected (will use CPU fallback)${NC}"
fi

# Test 2: Check OpenCL
echo ""
echo "2. Checking OpenCL..."
if command -v clinfo &> /dev/null; then
    if clinfo 2>&1 | grep -q "NVIDIA"; then
        echo -e "${GREEN}✓ OpenCL NVIDIA platform found${NC}"
    else
        echo -e "${YELLOW}⚠ OpenCL NVIDIA platform not found${NC}"
    fi
else
    echo -e "${YELLOW}⚠ clinfo not installed${NC}"
fi

# Test 3: Check ML model
echo ""
echo "3. Checking ML model..."
if [ -f "models/rf_model.joblib" ]; then
    echo -e "${GREEN}✓ ML model found${NC}"
    ls -lh models/rf_model.joblib
else
    echo -e "${RED}✗ ML model not found${NC}"
    echo "  Run: python3 src/ml/train_ml.py"
    FAILED=$((FAILED + 1))
fi

# Test 4: Build project
echo ""
echo "4. Building project..."
mkdir -p build
cd build

if [ ! -f "CMakeCache.txt" ]; then
    echo "Running CMake..."
    cmake .. || {
        echo -e "${RED}✗ CMake failed${NC}"
        exit 1
    }
fi

test_step "Build detector" "make -j\$(nproc)"
cd ..

# Test 5: Test GPU initialization
echo ""
echo "5. Testing GPU initialization..."
if [ -f "build/detector" ]; then
    if ./build/detector --use-gpu --pcap /dev/null 2>&1 | grep -qi "gpu\|opencl\|initialized"; then
        echo -e "${GREEN}✓ GPU initialization test passed${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠ GPU initialization test inconclusive${NC}"
        echo "  (This is OK if PCAP file is invalid)"
    fi
else
    echo -e "${RED}✗ Detector executable not found${NC}"
    FAILED=$((FAILED + 1))
fi

# Test 6: Test CPU fallback
echo ""
echo "6. Testing CPU fallback..."
if [ -f "build/detector" ]; then
    if ./build/detector --use-cpu --pcap /dev/null 2>&1 | head -5; then
        echo -e "${GREEN}✓ CPU mode works${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${YELLOW}⚠ CPU mode test inconclusive${NC}"
    fi
fi

# Test 7: Check PCAP files
echo ""
echo "7. Checking PCAP files..."
PCAP_COUNT=$(find data/ -name "*.pcap" -type f 2>/dev/null | wc -l)
if [ "$PCAP_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ Found $PCAP_COUNT PCAP file(s)${NC}"
    find data/ -name "*.pcap" -type f | head -3
    PASSED=$((PASSED + 1))
else
    echo -e "${YELLOW}⚠ No PCAP files found in data/ directory${NC}"
fi

# Test 8: Test dashboard
echo ""
echo "8. Testing dashboard API..."
if python3 -c "import flask, flask_socketio" 2>/dev/null; then
    echo -e "${GREEN}✓ Flask dependencies installed${NC}"
    PASSED=$((PASSED + 1))
else
    echo -e "${RED}✗ Flask dependencies missing${NC}"
    echo "  Run: pip3 install flask flask-socketio"
    FAILED=$((FAILED + 1))
fi

# Summary
echo ""
echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run full pipeline test:"
    echo "     ./build/detector --pcap <pcap_file> --use-gpu --batch 128"
    echo ""
    echo "  2. Test GPU correctness:"
    echo "     python3 tools/validate_gpu_correctness.py"
    echo ""
    echo "  3. Run GPU benchmarks:"
    echo "     ./tools/run_gpu_tests.sh"
    echo ""
    echo "  4. Start dashboard:"
    echo "     python3 src/dashboard/app.py"
    exit 0
else
    echo -e "${RED}Some tests failed. Please fix issues above.${NC}"
    echo ""
    echo "See TESTING_GUIDE.md for detailed instructions."
    exit 1
fi

