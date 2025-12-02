# DDoS Detection System Implementation Plan

## Project Overview

Build a high-performance DDoS detection and mitigation system using OpenCL on NVIDIA RTX 3060 GPU. The system will implement three detection algorithms (Entropy-based, ML-based, CUSUM/PCA), two blocking methods (RTBH + iptables simulation), and a Flask dashboard for real-time monitoring.

## System Architecture

```
Packet Replay (tcpreplay) 
    ↓
Feature Extraction (CPU preprocessing)
    ↓
Batch Buffer → GPU OpenCL Kernels (Entropy + Feature Extraction)
    ↓                    ↓
CPU ML Inference ←─── Feature Vectors
    ↓                    ↓
CUSUM/PCA Detection (CPU)
    ↓
Decision Engine (Combine all 3 detectors)
    ↓
RTBH Controller + iptables Simulator
    ↓
Flask Dashboard (Live metrics & graphs)
```

## Phase 1: Environment Setup & Dataset Acquisition (Days 1-3)

### 1.1 Install Dependencies

- **System packages**: `tcpreplay`, `tshark`, `scapy`, `build-essential`, `ocl-icd-opencl-dev`, `nvidia-opencl-dev`
- **Python packages**: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `flask`, `flask-socketio`, `psutil`, `joblib`, `scapy`
- **C++ libraries**: OpenCL headers, libpcap, Boost (for async operations)

### 1.2 Download Datasets to `data/` directory

**CIC-DDoS2019 Dataset:**

- Primary dataset from Mendeley Data
- Manual download required: https://data.mendeley.com/datasets/ssnc74xm6r/1
- Extract to `data/cic-ddos2019/`
- Contains labeled pcap files and CSV flow data

**CAIDA DDoS 2007 Dataset:**

- Requires approval via request form
- After approval, download and extract to `data/caida-ddos2007/`
- Contains anonymized pcap traces

**MAWI Traffic Archive (Optional):**

- Download sample traces using wget:
```bash
wget -P data/mawi/ http://mawi.wide.ad.jp/mawi/samplepoint-F/YYYY/MM/YYYYMMDD-HHMMSS.pcap.gz
```

- Extract to `data/mawi/` for background traffic

### 1.3 Verify GPU Setup

- Run `clinfo` to verify NVIDIA RTX 3060 is detected
- Test OpenCL with simple kernel to confirm GPU access
- Document GPU specs (compute units, memory, etc.)

**Deliverables:**

- `data/README.md` with dataset sources and structure
- `scripts/setup_environment.sh` installation script
- `scripts/prepare_data.py` for dataset preprocessing and labeling

## Phase 2: Baseline CPU Implementation (Days 4-10)

### 2.1 Packet Ingestion & Windowing

- **File**: `src/ingest/pcap_reader.cpp` (C++ wrapper around libpcap)
- **File**: `src/ingest/window_manager.cpp` - Tumbling windows (1s and 5s)
- Parse packets, extract 5-tuple flows
- Output windowed CSV: `window_start, src_ip, dst_ip, src_port, dst_port, protocol, pkt_count, byte_count, flow_count`

### 2.2 CPU Entropy Baseline

- **File**: `src/detectors/entropy_cpu.cpp`
- Calculate entropy for: Source IP, Destination IP, Source Port, Packet Size, Protocol
- Formula: `H = -Σ p_i * log2(p_i)`
- Validate with known attack traces, plot entropy over time
- **File**: `tools/validate_entropy.py` - Compare against ground truth

### 2.3 CPU CUSUM Implementation

- **File**: `src/detectors/cusum_detector.cpp`
- Implement CUSUM for change detection in entropy values
- Track cumulative sum of deviations from baseline
- Threshold-based alerting

### 2.4 CPU PCA Implementation

- **File**: `src/detectors/pca_detector.cpp`
- Principal Component Analysis for dimensionality reduction
- Anomaly detection using reconstruction error
- Use Eigen library or implement basic PCA

**Deliverables:**

- Working CPU baseline for all three detectors
- Validation scripts showing detection on labeled data
- Baseline performance metrics

## Phase 3: ML Pipeline - Training (Days 11-16)

### 3.1 Feature Engineering

- **File**: `src/ml/feature_extractor.py`
- Extract features per window:
  - Total packets, bytes, flows
  - Unique src/dst IP counts
  - Top-N IP fractions
  - Average packet size
  - Entropy features (from CPU baseline)
  - Flow duration statistics
- **File**: `src/ml/features_spec.json` - Feature specification

### 3.2 Model Training

- **File**: `src/ml/train_ml.py`
- Train Random Forest: `n_estimators=100, max_depth=10`
- Train on CIC-DDoS2019 labeled windows
- 5-fold cross-validation
- Export model: `models/rf_model.joblib`
- Generate ROC/PR curves: `results/ml_roc_curve.png`

### 3.3 Model Evaluation

- Test on hold-out set
- Compute Precision, Recall, F1, FPR
- Compare performance across attack types
- **File**: `src/ml/evaluate_model.py`

**Deliverables:**

- Trained Random Forest model (`models/rf_model.joblib`)
- Feature extraction pipeline
- Evaluation metrics and plots

## Phase 4: OpenCL GPU Implementation (Days 17-28)

### 4.1 OpenCL Host Setup (C++ with C-compatible OpenCL calls)

- **File**: `src/opencl/host.cpp`
- Device selection (NVIDIA RTX 3060)
- Context and command queue creation with profiling
- Buffer management (pinned memory for faster transfers)
- Event profiling for kernel timing

### 4.2 GPU Entropy Kernel

- **File**: `src/opencl/kernels/entropy.cl`
- Parallel entropy computation for multiple windows
- Kernel signature: `compute_entropy(counts, window_totals, entropy_out)`
- Process batches of 256+ windows simultaneously
- Optimize for RTX 3060 (warp size 32, shared memory usage)

### 4.3 GPU Feature Extraction Kernel

- **File**: `src/opencl/kernels/feature_extract.cl`
- Parallel histogram building for IP addresses
- Packet size distribution computation
- Port distribution analysis
- Minimize CPU-GPU transfers by batching

### 4.4 GPU Integration

- **File**: `src/opencl/gpu_detector.cpp`
- Integrate entropy and feature extraction kernels
- Batch processing: collect 128-256 windows before GPU transfer
- Async memory transfers with `CL_FALSE` flag
- Pipeline: Transfer → Compute → Transfer (overlap operations)

### 4.5 Correctness Validation

- **File**: `tools/validate_gpu_correctness.py`
- Compare GPU entropy output vs CPU baseline
- Tolerance: 1e-5 for floating point differences
- Performance comparison: GPU vs CPU speedup

**Deliverables:**

- Working OpenCL kernels for entropy and feature extraction
- GPU-accelerated detector with profiling
- Validation showing GPU correctness and speedup

## Phase 5: ML Inference Integration (Days 29-32)

### 5.1 Batched ML Inference (CPU)

- **File**: `src/ml/inference_engine.cpp` (C++ with Python bindings)
- Load Random Forest model using joblib
- Batch inference on GPU-extracted features
- Use Python C API or pybind11 for model loading
- Process batches of 128+ windows for efficiency

### 5.2 Decision Engine

- **File**: `src/detectors/decision_engine.cpp`
- Combine three detectors:
  - GPU Entropy scores
  - ML Random Forest probabilities
  - CUSUM/PCA anomaly scores
- Ensemble rule: Alert if (entropy > threshold) OR (RF > threshold) OR (CUSUM/PCA > threshold)
- Weighted scoring option: `score = w_e*entropy + w_m*RF + w_c*CUSUM`
- **File**: `config/detection_config.json` - Thresholds and weights

**Deliverables:**

- Integrated detection pipeline (GPU + CPU)
- Decision engine combining all three algorithms
- Alert generation system

## Phase 6: Blocking Implementation (Days 33-38)

### 6.1 RTBH Controller

- **File**: `src/blocking/rtbh_controller.cpp`
- Maintain blackhole list (JSON file: `blackhole.json`)
- Receive alerts from decision engine
- Add/remove IPs to blackhole list
- Log blocking actions: `logs/blocking.csv`

### 6.2 RTBH Blocker (pcap filtering)

- **File**: `src/blocking/pcap_filter.cpp`
- Read blackhole list periodically
- Filter packets during replay using libpcap/scapy
- Drop packets matching blackholed IPs
- Statistics: packets dropped, bytes dropped

### 6.3 iptables Simulation

- **File**: `src/blocking/iptables_simulator.cpp`
- Simulate iptables rule generation
- Create rule files: `iptables_rules.txt`
- Format: `iptables -A INPUT -s <IP> -j DROP`
- Optionally execute rules if running with sudo (with safety checks)
- Log simulated blocking actions

### 6.4 Blocking Metrics

- Track: % attack packets dropped, % legitimate packets dropped (collateral damage)
- Measure blocking latency (time from alert to rule application)
- **File**: `src/blocking/blocking_metrics.cpp`

**Deliverables:**

- RTBH controller and blocker
- iptables simulator
- Blocking effectiveness metrics

## Phase 7: Flask Dashboard (Days 39-42)

### 7.1 Backend API

- **File**: `src/dashboard/app.py` (Flask)
- REST API endpoints:
  - `/api/metrics` - Current system metrics
  - `/api/alerts` - Recent alerts
  - `/api/blackhole` - Current blackhole list
  - `/api/stats` - Detection statistics
- WebSocket support for live updates (`flask-socketio`)

### 7.2 Frontend Dashboard

- **File**: `src/dashboard/templates/index.html`
- **File**: `src/dashboard/static/js/dashboard.js`
- Real-time graphs (Chart.js or Plotly):
  - GPU utilization over time
  - Throughput (pps, Gbps)
  - Detection alerts timeline
  - Entropy values over time
  - ML confidence scores
  - Blocking effectiveness
- Toggle RTBH on/off button
- Display current blackhole list
- System status indicators

### 7.3 Data Collection for Dashboard

- **File**: `src/utils/metrics_collector.cpp`
- Collect: CPU%, GPU%, memory, pps_in, pps_processed, kernel_times
- Write to shared memory or file for Flask to read
- Update frequency: 1 second

**Deliverables:**

- Flask dashboard with live graphs
- Real-time metrics visualization
- Interactive blocking controls

## Phase 8: Performance Instrumentation (Days 43-45)

### 8.1 Timing Instrumentation

- Add timers throughout pipeline:
  - Ingestion → Feature Extraction → GPU Transfer → Kernel Execution → ML Inference → Decision → Blocking
- Use OpenCL event profiling for GPU kernels
- High-resolution timestamps (nanoseconds)

### 8.2 Resource Monitoring

- **File**: `src/utils/resource_monitor.cpp`
- CPU usage (per core)
- GPU utilization (via NVML or OpenCL queries)
- Memory usage (host and device)
- Network throughput

### 8.3 Logging System

- **File**: `src/utils/logger.cpp`
- Structured logging: CSV format
- Logs:
  - `logs/alerts.csv`: timestamp, window_start, src_ip, score, detector_type
  - `logs/metrics.csv`: timestamp, cpu%, gpu%, memMB, pps_in, pps_processed
  - `logs/blocking.csv`: timestamp, blackhole_applied, impacted_packets, dropped_packets
  - `logs/kernel_times.csv`: timestamp, kernel_name, execution_time_ms

**Deliverables:**

- Comprehensive instrumentation
- Detailed logging system
- Performance profiling data

## Phase 9: Experiment Scripts & Evaluation (Days 46-52)

### 9.1 Experiment Automation

- **File**: `scripts/run_experiment.sh`
- Arguments: `--pcap <file> --pps <rate> --window <secs> --batch <size> --model <rf>`
- Orchestrate: metrics collector → blocker → detector → replay → plotter
- Clean shutdown and log collection

### 9.2 Evaluation Scripts

- **File**: `scripts/evaluate_detection.py`
- Compute: Precision, Recall, F1, FPR, TPR
- Detection lead time (attack start to first alert)
- Per-attack-type analysis

### 9.3 Plotting Scripts

- **File**: `scripts/plot_results.py`
- Generate plots:
  - Detection lead time vs packet rate
  - Throughput vs batch size
  - TPR/FPR curves
  - Blocking effectiveness
  - GPU utilization timeline
  - Latency distribution (avg, 95th percentile)
  - Scalability analysis

### 9.4 Experiments to Run

1. **Correctness**: GPU vs CPU entropy comparison
2. **Accuracy**: Detection metrics on labeled dataset
3. **Lead Time**: Attack start to alert time
4. **Throughput**: Vary rates (100k, 200k, 500k, 1M pps)
5. **Latency**: Per-window processing time
6. **Resource Utilization**: CPU/GPU usage over time
7. **Blocking Effectiveness**: % attack dropped, collateral damage
8. **Ablation**: Entropy-only vs ML-only vs Combined vs All-three

**Deliverables:**

- Automated experiment scripts
- Evaluation metrics computation
- Comprehensive plots and tables

## Phase 10: Documentation & Report (Days 53-56)

### 10.1 Code Documentation

- Doxygen comments for C++ code
- Docstrings for Python code
- README files in each module directory

### 10.2 User Guide

- **File**: `README.md`
- Installation instructions
- Usage examples
- Configuration options
- Troubleshooting guide

### 10.3 Technical Report (10-12 pages)

- Abstract & CCP justification
- System architecture
- Detection algorithms (entropy math, RF hyperparameters, CUSUM/PCA)
- OpenCL kernel design
- RTBH + iptables architecture
- Experimental setup (hardware: RTX 3060 specs, software versions)
- Results (all required metrics with tables/plots)
- Discussion (trade-offs, limitations)
- Conclusion & future work

### 10.4 Presentation Slides (10-12 slides)

- System overview
- Architecture diagram
- Detection algorithms
- GPU acceleration highlights
- Key results
- Demo screenshots

**Deliverables:**

- Complete documentation
- Technical report
- Presentation slides

## Key Files Structure

```
project-root/
├── data/                          # Datasets
│   ├── cic-ddos2019/
│   ├── caida-ddos2007/
│   ├── mawi/
│   └── README.md
├── src/
│   ├── ingest/
│   │   ├── pcap_reader.cpp
│   │   └── window_manager.cpp
│   ├── opencl/
│   │   ├── host.cpp
│   │   ├── gpu_detector.cpp
│   │   └── kernels/
│   │       ├── entropy.cl
│   │       └── feature_extract.cl
│   ├── detectors/
│   │   ├── entropy_cpu.cpp
│   │   ├── cusum_detector.cpp
│   │   ├── pca_detector.cpp
│   │   └── decision_engine.cpp
│   ├── ml/
│   │   ├── feature_extractor.py
│   │   ├── train_ml.py
│   │   ├── inference_engine.cpp
│   │   └── features_spec.json
│   ├── blocking/
│   │   ├── rtbh_controller.cpp
│   │   ├── pcap_filter.cpp
│   │   └── iptables_simulator.cpp
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── templates/index.html
│   │   └── static/js/dashboard.js
│   └── utils/
│       ├── metrics_collector.cpp
│       ├── resource_monitor.cpp
│       └── logger.cpp
├── scripts/
│   ├── setup_environment.sh
│   ├── prepare_data.py
│   ├── run_experiment.sh
│   ├── evaluate_detection.py
│   └── plot_results.py
├── models/
│   └── rf_model.joblib
├── config/
│   └── detection_config.json
├── logs/                          # Generated during experiments
├── results/                       # Plots and analysis
├── tools/
│   ├── validate_entropy.py
│   └── validate_gpu_correctness.py
├── CMakeLists.txt                # Build system
├── README.md
└── report/                        # Final report and slides
```

## Critical Implementation Notes

### GPU Optimization Strategy

- **Maximize GPU usage**: Offload entropy computation, histogram building, and feature extraction to GPU
- **Batch processing**: Process 128-256 windows per batch to amortize transfer overhead
- **Async transfers**: Use `CL_FALSE` for non-blocking transfers, pipeline operations
- **Memory management**: Use pinned (page-locked) memory for faster CPU-GPU transfers
- **Kernel optimization**: Use local memory for reductions, optimize work-group sizes for RTX 3060

### ML Inference Strategy

- Keep ML inference on CPU (as per requirement 2b) but use batched inference
- Load model once, process batches of 128+ windows
- Use Python C API or pybind11 for efficient model loading

### Detection Algorithm Combination

- Three independent detectors: Entropy (GPU), ML (CPU), CUSUM/PCA (CPU)
- Ensemble decision: Alert if any detector exceeds threshold
- Report individual and combined performance

### Blocking Methods

- **RTBH**: JSON-based blackhole list, pcap filtering during replay
- **iptables**: Rule file generation, optional execution with safety checks
- Both methods log blocking actions and effectiveness

## Success Criteria

1. ✅ GPU handles majority of computational load (entropy + feature extraction)
2. ✅ Three detection algorithms implemented and working
3. ✅ Two blocking methods functional
4. ✅ Flask dashboard with live graphs operational
5. ✅ All required metrics measured and reported
6. ✅ System processes 200k+ pps on RTX 3060
7. ✅ Detection lead time < 5 seconds for attacks
8. ✅ Accuracy metrics: F1 > 0.85, FPR < 0.05

## Timeline Summary

- **Weeks 1-2**: Setup, datasets, CPU baselines
- **Weeks 3-4**: ML training, OpenCL GPU implementation
- **Weeks 5-6**: Integration, blocking, dashboard
- **Weeks 7-8**: Experiments, evaluation, documentation

Total: ~8 weeks of focused development