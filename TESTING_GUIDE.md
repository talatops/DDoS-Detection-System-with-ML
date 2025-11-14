# Testing Guide for DDoS Detection System

This guide explains how to test the entire system at different levels.

## Table of Contents
1. [Testing Levels](#testing-levels)
2. [Unit Testing](#unit-testing)
3. [Component Testing](#component-testing)
4. [Integration Testing](#integration-testing)
5. [End-to-End Testing](#end-to-end-testing)
6. [Performance Testing](#performance-testing)

---

## Testing Levels

### 1. **Unit Testing** (Individual Functions/Classes)
- Test individual components in isolation
- Fast, no dependencies

### 2. **Component Testing** (Single Component with Mock Data)
- Test PCAP reader with sample packets
- Test ML model with CSV rows
- Test detectors with synthetic windows

### 3. **Integration Testing** (Multiple Components Together)
- Test PCAP → Windows → Features → Detection pipeline
- Test GPU kernels with real data

### 4. **End-to-End Testing** (Full System)
- Test with real PCAP files
- Test blocking mechanisms
- Test dashboard updates

### 5. **Performance Testing** (Throughput, Latency)
- Test high packet rates
- Measure GPU utilization
- Benchmark detection latency

---

## Unit Testing

### Test PCAP Reader
```bash
# Create a simple test script
cat > test_pcap_reader.cpp << 'EOF'
#include "ingest/pcap_reader.h"
#include <iostream>

int main() {
    PcapReader reader;
    if (!reader.open("data/test_small.pcap")) {
        std::cerr << "Failed to open PCAP" << std::endl;
        return 1;
    }
    
    PacketInfo packet;
    int count = 0;
    while (reader.getNextPacket(packet) && count < 10) {
        std::cout << "Packet " << count << ": " 
                  << packet.src_ip << " -> " << packet.dst_ip 
                  << " (" << packet.packet_len << " bytes)" << std::endl;
        count++;
    }
    return 0;
}
EOF

# Compile and run
g++ -o test_pcap_reader test_pcap_reader.cpp src/ingest/pcap_reader.cpp -lpcap
./test_pcap_reader
```

### Test ML Model (Python)
```bash
# Test with a single CSV row
python3 << 'EOF'
import pandas as pd
import joblib
import sys
sys.path.insert(0, 'src/ml')
from inference_engine import MLInferenceEngine

# Load model
engine = MLInferenceEngine()
engine.loadModel("models/rf_model.joblib")

# Create test feature vector (16 features matching your model)
test_features = [
    1000,      # total_packets
    50000,     # total_bytes
    50,        # unique_src_ips
    1,         # unique_dst_ips
    100,       # unique_src_ports
    1,         # unique_dst_ports
    50,        # flow_count
    50.0,      # avg_packet_size
    3.5,       # src_ip_entropy
    0.0,       # dst_ip_entropy
    4.2,       # src_port_entropy
    0.0,       # dst_port_entropy
    2.1,       # packet_size_entropy
    1.5,       # protocol_entropy
    0.8,       # top10_src_ip_fraction
    1.0        # top10_dst_ip_fraction
]

# Predict
score = engine.predict(test_features)
print(f"Attack probability: {score:.4f}")
print(f"Prediction: {'ATTACK' if score > 0.5 else 'BENIGN'}")
EOF
```

### Test Window Manager
```bash
# Create test with synthetic packets
cat > test_window.cpp << 'EOF'
#include "ingest/window_manager.h"
#include <iostream>

int main() {
    WindowManager wm(1); // 1 second window
    
    // Create test packets
    PacketInfo p1, p2, p3;
    p1.timestamp_us = 0;
    p1.src_ip = "192.168.1.1";
    p1.dst_ip = "10.0.0.1";
    p1.packet_len = 100;
    
    p2.timestamp_us = 500000; // 0.5 seconds
    p2.src_ip = "192.168.1.2";
    p2.dst_ip = "10.0.0.1";
    p2.packet_len = 200;
    
    p3.timestamp_us = 1500000; // 1.5 seconds (new window)
    p3.src_ip = "192.168.1.1";
    p3.dst_ip = "10.0.0.1";
    p3.packet_len = 150;
    
    wm.addPacket(p1);
    wm.addPacket(p2);
    wm.addPacket(p3);
    
    auto windows = wm.getCompletedWindows();
    std::cout << "Completed windows: " << windows.size() << std::endl;
    
    return 0;
}
EOF
```

---

## Component Testing

### Test PCAP Reading with Real File
```bash
# Extract first 100 packets from a PCAP for quick testing
editcap -c 100 data/cic-ddos2019/ddostrace.20070804_145436.pcap \
        data/test_small.pcap

# Test PCAP reader
./bin/detector --pcap data/test_small.pcap --window 1 --batch 32 --test-mode
```

### Test ML Model with CSV Data
```bash
# Test with a single row from CSV
python3 << 'EOF'
import pandas as pd
import joblib
import sys
sys.path.insert(0, 'src/ml')
from inference_engine import MLInferenceEngine
from feature_extractor import FeatureExtractor

# Load a CSV file
df = pd.read_csv("data/caida-ddos2007/sample.csv", nrows=1)

# Extract features (using your feature extraction logic)
extractor = FeatureExtractor()
features = extractor.extract_features_from_csv(df.iloc[0])

# Load model and predict
engine = MLInferenceEngine()
engine.loadModel("models/rf_model.joblib")
score = engine.predict(features.tolist())

print(f"Features: {features}")
print(f"Attack probability: {score:.4f}")
EOF
```

### Test GPU Entropy Calculation
```bash
# Create a test OpenCL program
cat > test_gpu_entropy.cpp << 'EOF'
#include "opencl/gpu_detector.h"
#include "ingest/window_manager.h"
#include <iostream>

int main() {
    GpuDetector gpu;
    if (!gpu.initialize()) {
        std::cerr << "GPU initialization failed" << std::endl;
        return 1;
    }
    
    // Create test window with known entropy
    WindowFeatures window;
    window.packet_count = 100;
    window.src_ip_counts["192.168.1.1"] = 50;
    window.src_ip_counts["192.168.1.2"] = 30;
    window.src_ip_counts["192.168.1.3"] = 20;
    
    auto entropy = gpu.computeEntropy(window);
    std::cout << "GPU Entropy: " << entropy.src_ip_entropy << std::endl;
    
    return 0;
}
EOF
```

---

## Integration Testing

### Test Full Pipeline (PCAP → Detection)
```bash
# Use a small PCAP file
./bin/detector \
    --pcap data/test_small.pcap \
    --window 1 \
    --batch 128 \
    --model models/rf_model.joblib \
    --output logs/test_integration \
    --verbose
```

### Test with Ground Truth Labels
```bash
# If you have labeled PCAP files
./bin/detector \
    --pcap data/cic-ddos2019/attack_trace.pcap \
    --ground-truth data/cic-ddos2019/labels.csv \
    --window 1 \
    --model models/rf_model.joblib \
    --output logs/test_with_labels
```

---

## End-to-End Testing

### Test 1: Small PCAP File (Quick Test)
```bash
# Step 1: Create a small test PCAP (first 1000 packets)
editcap -c 1000 data/cic-ddos2019/ddostrace.20070804_145436.pcap \
        data/test_small.pcap

# Step 2: Run detector
./bin/detector \
    --pcap data/test_small.pcap \
    --window 1 \
    --batch 64 \
    --model models/rf_model.joblib \
    --output logs/test_small

# Step 3: Check results
ls -lh logs/test_small/
cat logs/test_small/alerts.csv | head -20
```

### Test 2: Medium PCAP File (Realistic Test)
```bash
# Use a medium-sized PCAP (10,000 packets)
editcap -c 10000 data/cic-ddos2019/ddostrace.20070804_145436.pcap \
        data/test_medium.pcap

# Run with experiment script
./scripts/run_experiment.sh \
    --pcap data/test_medium.pcap \
    --pps 50000 \
    --window 1 \
    --batch 128

# Evaluate results
python3 scripts/evaluate_detection.py logs/exp_*/
```

### Test 3: Full Dataset (Complete Test)
```bash
# Run on full PCAP file
./scripts/run_experiment.sh \
    --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap \
    --pps 200000 \
    --window 1 \
    --batch 256 \
    --output logs/full_test

# Generate plots
python3 scripts/plot_results.py logs/full_test
```

### Test 4: With Dashboard (Visual Testing)
```bash
# Terminal 1: Start dashboard
python3 src/dashboard/app.py

# Terminal 2: Run detector (it will send metrics to dashboard)
./bin/detector \
    --pcap data/test_medium.pcap \
    --window 1 \
    --dashboard-url http://localhost:5000 \
    --model models/rf_model.joblib

# Open browser: http://localhost:5000
# Watch live metrics, alerts, GPU utilization
```

---

## Performance Testing

### Test Throughput
```bash
# Test with high packet rate
./scripts/run_experiment.sh \
    --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap \
    --pps 500000 \
    --window 1 \
    --batch 256

# Check logs for throughput metrics
grep "packets_per_second" logs/exp_*/metrics.csv
```

### Test GPU Utilization
```bash
# Run detector and monitor GPU
watch -n 1 nvidia-smi

# In another terminal
./bin/detector --pcap data/test_medium.pcap --window 1 --batch 256
```

### Benchmark Detection Latency
```bash
# Run with timing enabled
./bin/detector \
    --pcap data/test_small.pcap \
    --window 1 \
    --batch 128 \
    --profile \
    --output logs/benchmark

# Check timing logs
cat logs/benchmark/timings.csv
```

---

## Testing Workflow Summary

### Quick Test (5 minutes)
```bash
# 1. Test ML model with CSV row
python3 scripts/test_ml_single.py data/caida-ddos2007/sample.csv

# 2. Test PCAP reading
./bin/detector --pcap data/test_small.pcap --test-mode

# 3. Verify GPU works
clinfo | grep "Device"
```

### Standard Test (30 minutes)
```bash
# 1. Run on medium PCAP
./scripts/run_experiment.sh --pcap data/test_medium.pcap --pps 100000

# 2. Evaluate results
python3 scripts/evaluate_detection.py logs/exp_*/

# 3. Check plots
python3 scripts/plot_results.py logs/exp_*/
```

### Full Test (2+ hours)
```bash
# 1. Run on full dataset
./scripts/run_experiment.sh --pcap data/cic-ddos2019/full_trace.pcap

# 2. Run with dashboard
python3 src/dashboard/app.py &
./bin/detector --pcap data/full_trace.pcap --dashboard-url http://localhost:5000

# 3. Comprehensive evaluation
python3 scripts/evaluate_detection.py logs/exp_*/
python3 scripts/plot_results.py logs/exp_*/
```

---

## Expected Outputs

### Successful Test Should Show:
1. **PCAP Reading**: "Processed X packets"
2. **Window Creation**: "Created Y windows"
3. **GPU Detection**: "GPU entropy computed: X.XX"
4. **ML Prediction**: "Attack probability: X.XX"
5. **Alerts**: "Alert: DDoS detected at timestamp X"
6. **Blocking**: "Blocked IP: X.X.X.X"
7. **Metrics**: CSV files with timing, throughput, accuracy

### Log Files Generated:
- `logs/exp_*/alerts.csv` - All detection alerts
- `logs/exp_*/metrics.csv` - Performance metrics
- `logs/exp_*/timings.csv` - Component timings
- `logs/exp_*/gpu_utilization.csv` - GPU usage
- `logs/exp_*/blocking_actions.csv` - Blocking events

---

## Troubleshooting Tests

### If PCAP reading fails:
```bash
# Check file format
file data/test.pcap
# Should show: "pcap capture file"

# Check file permissions
ls -l data/test.pcap
```

### If ML model fails:
```bash
# Verify model exists
ls -lh models/rf_model.joblib

# Test model loading
python3 -c "import joblib; m=joblib.load('models/rf_model.joblib'); print('OK')"
```

### If GPU fails:
```bash
# Check OpenCL
clinfo | grep "Platform"

# Check GPU
nvidia-smi
```

---

## Next Steps After Testing

1. **If tests pass**: Proceed to full dataset evaluation
2. **If tests fail**: Check logs, fix issues, retest
3. **Performance issues**: Optimize batch sizes, window sizes
4. **Accuracy issues**: Retrain ML model, adjust thresholds

