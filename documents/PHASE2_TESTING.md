# Phase 2: Validation and Testing Guide

## Overview

This document describes the comprehensive validation and testing suite for Phase 2 (Baseline CPU Implementation) components.

## Test Components

### 1. Entropy Validation (`tools/validate_entropy.py`)

**Purpose**: Validate entropy calculation correctness and DDoS attack pattern detection.

**Tests**:
- ✅ Entropy formula correctness (uniform distribution, single value, empty counts)
- ✅ DDoS attack pattern detection (low entropy vs normal traffic)
- ✅ Entropy calculation from CSV data
- ✅ Entropy plots over time

**Usage**:
```bash
python3 tools/validate_entropy.py
```

**Output**:
- Console output with test results
- `results/entropy_validation.png` - Entropy plots over time

### 2. C++ Unit Tests (`tools/test_phase2.cpp`)

**Purpose**: Comprehensive unit tests for all Phase 2 C++ components.

**Tests**:
- ✅ Entropy calculation (uniform, single value, empty)
- ✅ Window manager (packet addition, window closure)
- ✅ CUSUM detector (baseline, anomaly detection, reset)
- ✅ PCA detector (training, normal/attack detection)
- ✅ Integration test (full pipeline)

**Compilation**:
```bash
g++ -o build/test_phase2 tools/test_phase2.cpp -I. -std=c++11 -lpcap
```

**Usage**:
```bash
./build/test_phase2
```

### 3. PCAP Processing Test (`tools/test_pcap_processing.py`)

**Purpose**: Validate PCAP file reading and window processing.

**Tests**:
- ✅ PCAP file availability check
- ✅ Build verification
- ✅ PCAP reading and packet parsing
- ✅ Window creation and statistics

**Usage**:
```bash
python3 tools/test_pcap_processing.py
```

**Requirements**:
- PCAP files in `data/cic-ddos2019/` or `data/caida-ddos2007/`
- Compiled C++ project in `build/`

### 4. Performance Benchmarks (`tools/benchmark_phase2.py`)

**Purpose**: Measure performance characteristics of Phase 2 components.

**Benchmarks**:
- ✅ Entropy calculation throughput (various histogram sizes)
- ✅ Window processing speed (various packet rates)
- ✅ Memory usage (various window sizes)

**Usage**:
```bash
python3 tools/benchmark_phase2.py
```

**Output**:
- Console output with benchmark results
- `results/phase2_benchmarks.json` - Detailed benchmark data

### 5. Comprehensive Test Runner (`tools/run_phase2_tests.sh`)

**Purpose**: Run all Phase 2 tests in sequence.

**Usage**:
```bash
./tools/run_phase2_tests.sh
```

**What it does**:
1. Runs entropy validation
2. Builds C++ project
3. Runs C++ unit tests (if available)
4. Runs PCAP processing tests
5. Runs performance benchmarks
6. Provides summary report

## Expected Results

### Entropy Validation
- ✅ All entropy formula tests pass
- ✅ DDoS attack pattern correctly identified (low entropy)
- ✅ Entropy plots generated successfully

### C++ Unit Tests
- ✅ All 5 test suites pass
- ✅ No assertion failures
- ✅ Integration test completes successfully

### Performance Benchmarks
- ✅ Entropy calculation: >1000 calc/sec
- ✅ Window processing: >100k packets/sec
- ✅ Memory efficiency: <100 bytes/packet

## Troubleshooting

### Issue: "No PCAP files found"
**Solution**: Download datasets to `data/` directory:
- CIC-DDoS2019: `data/cic-ddos2019/`
- CAIDA DDoS 2007: `data/caida-ddos2007/`

### Issue: "Build failed"
**Solution**: Install dependencies and rebuild:
```bash
sudo apt-get install libpcap-dev build-essential cmake
mkdir -p build && cd build && cmake .. && make
```

### Issue: "Test executable not found"
**Solution**: Compile test programs:
```bash
g++ -o build/test_phase2 tools/test_phase2.cpp -I. -std=c++11 -lpcap
```

## Next Steps

After Phase 2 validation passes:
1. ✅ Proceed to Phase 4 (OpenCL GPU Implementation)
2. ✅ Compare GPU vs CPU performance
3. ✅ Validate GPU correctness against CPU baseline

## Files Created

- `tools/validate_entropy.py` - Entropy validation script
- `tools/test_phase2.cpp` - C++ unit tests
- `tools/test_pcap_processing.py` - PCAP processing test
- `tools/benchmark_phase2.py` - Performance benchmarks
- `tools/run_phase2_tests.sh` - Comprehensive test runner
- `PHASE2_TESTING.md` - This document

