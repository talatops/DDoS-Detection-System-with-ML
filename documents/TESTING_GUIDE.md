# Testing Guide

This guide covers how to test the DDoS detection system after Phase 4 and Phase 5 implementation.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Building the Project](#building-the-project)
3. [Quick Tests](#quick-tests)
4. [GPU Testing](#gpu-testing)
5. [ML Inference Testing](#ml-inference-testing)
6. [Full Pipeline Testing](#full-pipeline-testing)
7. [Dashboard Testing](#dashboard-testing)
8. [Troubleshooting](#troubleshooting)

---

## Prerequisites

Before testing, ensure:

1. **GPU Setup**:
   ```bash
   # Check GPU is detected
   nvidia-smi
   
   # Check OpenCL is working
   clinfo | grep -A 5 "NVIDIA"
   ```

2. **ML Model Trained**:
   ```bash
   # Check if model exists
   ls -lh models/rf_model.joblib
   ls -lh models/preprocessor.joblib
   
   # If not, train it:
   python3 src/ml/train_ml.py
   ```

3. **Dependencies Installed**:
   ```bash
   # Run setup script if needed
   ./scripts/setup_environment.sh
   ```

---

## Building the Project

### Step 1: Configure and Build

```bash
cd /home/talatfaheem/PDC/project

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake ..

# Build (use all CPU cores)
make -j$(nproc)
```

**Expected Output:**
- `build/detector` executable created
- No compilation errors

**Common Issues:**
- **Missing OpenCL**: Install `ocl-icd-opencl-dev` and `nvidia-opencl-icd`
- **Missing Python**: Install `python3-dev` and `python3-numpy`
- **Missing libpcap**: Install `libpcap-dev`

### Step 2: Verify Build

```bash
# Check executable exists
ls -lh build/detector

# Check it runs (should show help/usage)
./build/detector --help 2>&1 || ./build/detector 2>&1 | head -5
```

---

## Quick Tests

### Test 1: GPU Initialization

```bash
cd build
./detector --use-gpu --pcap /dev/null 2>&1 | grep -i "gpu\|opencl\|nvidia"
```

**Expected:** Should see "GPU detector initialized successfully" or "OpenCL initialized successfully"

### Test 2: CPU Fallback

```bash
cd build
./detector --use-cpu --pcap /dev/null 2>&1 | head -10
```

**Expected:** Should run without GPU (may fail on PCAP reading, but should initialize)

### Test 3: ML Model Loading

```bash
cd build
./detector --use-cpu --pcap /dev/null 2>&1 | grep -i "ml\|model"
```

**Expected:** Should see "ML model loaded successfully" or "Failed to load ML model"

---

## GPU Testing

### Test GPU Correctness

```bash
# Run GPU validation script
python3 tools/validate_gpu_correctness.py

# Expected output:
# - Comparison report in results/gpu_correctness_report.txt
# - Scatter plots in results/gpu_cpu_entropy_comparison.png
```

### Test GPU Performance

```bash
# Build GPU benchmark
g++ -std=c++17 -O3 -I. \
    tools/benchmark_gpu_performance.cpp \
    src/opencl/gpu_detector.cpp \
    src/opencl/host.cpp \
    src/detectors/entropy_cpu.cpp \
    src/ingest/window_manager.cpp \
    -lOpenCL -lpcap -lpthread \
    -o build/benchmark_gpu_performance

# Run benchmark
./build/benchmark_gpu_performance
```

**Expected Output:**
- CPU vs GPU timing comparison
- Speedup metrics (should show GPU faster for large batches)
- Kernel execution times

### Run All GPU Tests

```bash
# Comprehensive GPU test suite
./tools/run_gpu_tests.sh
```

---

## ML Inference Testing

### Test ML Model Loading (Python)

```bash
# Test ML inference engine directly
python3 -c "
from src.ml.inference_engine import MLInferenceEngine
import numpy as np

engine = MLInferenceEngine()
if engine.loadModel('models/rf_model.joblib'):
    # Create dummy feature vector (16 features)
    features = np.random.rand(16).tolist()
    prob = engine.predict(features)
    print(f'ML Prediction: {prob:.4f}')
else:
    print('Failed to load model')
"
```

### Test Feature Builder

```bash
# Test feature extraction
python3 -c "
import sys
sys.path.insert(0, 'src')
from ml.feature_builder import FeatureBuilder
from ingest.window_manager import WindowStats

# Create dummy window
window = WindowStats()
window.total_packets = 1000
window.total_bytes = 1000000
window.src_ip_counts = {1: 500, 2: 300, 3: 200}
window.dst_ip_counts = {100: 1000}
window.src_port_counts = {80: 800, 443: 200}
window.dst_port_counts = {80: 1000}
window.packet_size_counts = {64: 500, 1500: 500}
window.protocol_counts = {6: 600, 17: 400}

builder = FeatureBuilder()
gpu_entropy = [5.0, 4.5, 6.0, 5.5, 3.0, 2.0]  # 6 entropy values
features = []
if builder.buildFeatures(window, gpu_entropy, features):
    print(f'Features extracted: {len(features)} features')
    print(f'First 5: {features[:5]}')
else:
    print('Feature extraction failed')
"
```

---

## Full Pipeline Testing

### Test with Small PCAP File

```bash
# Find a small PCAP file (or create one)
# Option 1: Use existing PCAP
PCAP_FILE="data/cic-ddos2019/ddostrace.20070804_145436.pcap"

# Option 2: Create a small test PCAP (if you have tcpdump)
# tcpdump -i any -c 100 -w test.pcap

# Run detector with GPU
cd build
./detector \
    --pcap "$PCAP_FILE" \
    --window 1 \
    --batch 128 \
    --use-gpu \
    2>&1 | tee ../logs/detector_test.log

# Check output
grep -i "attack\|detected\|processed" ../logs/detector_test.log
```

**Expected Output:**
- "Processing packets..."
- "Processed X packets total"
- "Processed X windows total"
- "GPU kernel execution time: X ms" (if GPU used)
- Attack detections (if any)

### Test with CPU Fallback

```bash
cd build
./detector \
    --pcap "$PCAP_FILE" \
    --window 1 \
    --batch 128 \
    --use-cpu \
    2>&1 | tee ../logs/detector_test_cpu.log
```

### Test Different Batch Sizes

```bash
cd build
for batch in 32 64 128 256; do
    echo "Testing batch size: $batch"
    ./detector \
        --pcap "$PCAP_FILE" \
        --window 1 \
        --batch $batch \
        --use-gpu \
        2>&1 | grep -E "processed|kernel time"
done
```

---

## Dashboard Testing

### Step 1: Start Dashboard

```bash
# Terminal 1: Start dashboard
cd /home/talatfaheem/PDC/project
python3 src/dashboard/app.py
```

**Expected:** Dashboard starts on http://localhost:5000

### Step 2: Test API Endpoints

```bash
# Terminal 2: Test API endpoints

# Get metrics
curl http://localhost:5000/api/metrics

# Get alerts
curl http://localhost:5000/api/alerts

# Get blackhole list
curl http://localhost:5000/api/blackhole

# Get training metrics
curl http://localhost:5000/api/training-metrics

# Get stats
curl http://localhost:5000/api/stats
```

### Step 3: Send Test Alert (POST)

```bash
# Send a test alert to dashboard
curl -X POST http://localhost:5000/api/alerts \
    -H "Content-Type: application/json" \
    -d '{
        "timestamp": "2024-01-01T12:00:00",
        "window_start": "2024-01-01T12:00:00",
        "src_ip": "192.168.1.100",
        "entropy_score": 0.85,
        "ml_score": 0.92,
        "cusum_score": 4.5,
        "pca_score": 2.1,
        "detector_type": "entropy+ml",
        "is_attack": true
    }'
```

**Expected:** Returns `{"status": "success", "alert_id": 1}`

### Step 4: Send Test Blocking Update

```bash
# Send blocking update
curl -X POST http://localhost:5000/api/blocking \
    -H "Content-Type: application/json" \
    -d '{
        "ip": "192.168.1.100",
        "action": "add",
        "packets_dropped": 1000,
        "bytes_dropped": 1000000,
        "is_false_positive": false
    }'
```

### Step 5: Run Detector with Dashboard

```bash
# Terminal 2: Run detector with dashboard URL
cd build
./detector \
    --pcap "$PCAP_FILE" \
    --window 1 \
    --batch 128 \
    --use-gpu \
    --dashboard-url http://localhost:5000 \
    2>&1 | tee ../logs/detector_with_dashboard.log
```

**Note:** Currently, the C++ detector doesn't send alerts to dashboard automatically. This requires implementing the dashboard client (Phase 5.2).

---

## Integration Testing

### Test Complete Pipeline

```bash
# 1. Start dashboard
python3 src/dashboard/app.py &
DASHBOARD_PID=$!

# 2. Wait for dashboard to start
sleep 2

# 3. Run detector
cd build
./detector \
    --pcap "$PCAP_FILE" \
    --window 1 \
    --batch 256 \
    --use-gpu

# 4. Check logs
cd ..
ls -lh logs/
cat logs/alerts.csv | tail -10
cat logs/metrics.csv | tail -10

# 5. Stop dashboard
kill $DASHBOARD_PID
```

---

## Troubleshooting

### Issue: GPU Not Detected

```bash
# Check GPU
nvidia-smi

# Check OpenCL
clinfo | grep -i nvidia

# If OpenCL not found:
sudo apt-get install nvidia-opencl-icd
sudo ldconfig
```

### Issue: ML Model Not Found

```bash
# Check if model exists
ls -lh models/rf_model.joblib

# If not, train it:
python3 src/ml/train_ml.py
```

### Issue: Build Fails

```bash
# Clean and rebuild
cd build
rm -rf *
cmake ..
make -j$(nproc) 2>&1 | tee ../build_errors.log

# Check common errors:
grep -i "error\|undefined\|missing" ../build_errors.log
```

### Issue: PCAP File Not Found

```bash
# List available PCAP files
find data/ -name "*.pcap" -type f | head -5

# Or create a test PCAP:
tcpdump -i any -c 100 -w test.pcap
```

### Issue: Segmentation Fault

```bash
# Run with gdb for debugging
cd build
gdb ./detector
(gdb) run --pcap "$PCAP_FILE" --use-gpu
(gdb) bt  # Backtrace when it crashes
```

---

## Performance Testing

### Benchmark Throughput

```bash
# Test different packet rates
for rate in 100000 200000 500000 1000000; do
    echo "Testing rate: $rate pps"
    time ./build/detector \
        --pcap "$PCAP_FILE" \
        --window 1 \
        --batch 256 \
        --use-gpu \
        2>&1 | grep -E "processed|time"
done
```

### Monitor GPU Utilization

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run detector
cd build
./detector --pcap "$PCAP_FILE" --use-gpu --batch 256
```

---

## Expected Test Results

### GPU Tests
- ✅ GPU initialization: SUCCESS
- ✅ GPU entropy calculation: Within 1e-5 tolerance vs CPU
- ✅ GPU speedup: 2-10x faster than CPU (depends on batch size)
- ✅ GPU utilization: 50-90% during processing

### ML Tests
- ✅ Model loading: SUCCESS
- ✅ Feature extraction: 16 features per window
- ✅ Inference time: < 10ms per batch of 128 windows
- ✅ Predictions: Reasonable probabilities (0.0-1.0)

### Pipeline Tests
- ✅ Packet processing: No crashes
- ✅ Window creation: Correct window counts
- ✅ Detection: Alerts generated for attacks
- ✅ Logging: CSV files created in logs/

---

## Next Steps

After successful testing:

1. **Phase 6**: Integrate blocking (RTBH + iptables)
2. **Phase 8**: Add performance instrumentation
3. **Phase 9**: Create experiment automation scripts
4. **Phase 10**: Write technical report

See `REMAINING_PHASES_PLAN.md` for details.
