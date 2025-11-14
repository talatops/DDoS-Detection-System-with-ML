#!/bin/bash
# Comprehensive test runner for Phase 2 components

set -e

echo "================================================================================"
echo "Phase 2: Comprehensive Validation and Testing"
echo "================================================================================"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test results
PASSED=0
FAILED=0

# Function to run test and check result
run_test() {
    local test_name=$1
    local test_command=$2
    
    echo ""
    echo "Running: $test_name"
    echo "Command: $test_command"
    echo "--------------------------------------------------------------------------------"
    
    if eval "$test_command"; then
        echo -e "${GREEN}✓ PASSED: $test_name${NC}"
        ((PASSED++))
        return 0
    else
        echo -e "${RED}✗ FAILED: $test_name${NC}"
        ((FAILED++))
        return 1
    fi
}

# Test 1: Python entropy validation
run_test "Entropy Formula Validation" \
    "python3 tools/validate_entropy.py"

# Test 2: Build C++ project
if [ ! -d "build" ]; then
    echo "Creating build directory..."
    mkdir -p build
fi

if [ ! -f "build/CMakeCache.txt" ]; then
    echo "Running CMake..."
    cd build && cmake .. && cd ..
fi

run_test "Build C++ Project" \
    "cd build && make -j$(nproc) && cd .."

# Test 3: C++ unit tests (if test executable exists)
if [ -f "build/test_phase2" ]; then
    run_test "C++ Unit Tests" \
        "./build/test_phase2"
else
    echo -e "${YELLOW}⚠ SKIPPED: C++ unit tests (test_phase2 not found)${NC}"
    echo "  To compile: g++ -o build/test_phase2 tools/test_phase2.cpp -I. -std=c++11 -lpcap"
fi

# Test 4: PCAP processing test
run_test "PCAP Processing Test" \
    "python3 tools/test_pcap_processing.py"

# Test 5: Performance benchmarks
run_test "Performance Benchmarks" \
    "python3 tools/benchmark_phase2.py"

# Summary
echo ""
echo "================================================================================"
echo "Test Summary"
echo "================================================================================"
echo -e "${GREEN}Passed: $PASSED${NC}"
echo -e "${RED}Failed: $FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All Phase 2 tests PASSED!${NC}"
    exit 0
else
    echo -e "${RED}Some tests FAILED. Please review the output above.${NC}"
    exit 1
fi

