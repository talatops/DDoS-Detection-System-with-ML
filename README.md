# DDoS Detection System - GPU Implementation

High-performance DDoS detection and mitigation system using OpenCL on NVIDIA RTX 3060 GPU.

## Table of Contents

1. [Features](#features)
2. [System Architecture](#system-architecture)
3. [Project Status](#project-status)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Implementation Phases](#implementation-phases)
9. [Testing & Validation](#testing--validation)
10. [Configuration](#configuration)
11. [Documentation](#documentation)
12. [License](#license)

---

## Features

- **Three Detection Algorithms**:
  - Entropy-based detection (GPU-accelerated) - Real entropy calculation implemented
  - Machine Learning (Random Forest) - **99.59% accuracy** with real entropy features
  - CUSUM/PCA statistical detection

- **Two Blocking Methods**:
  - Remote Triggered Black Hole (RTBH)
  - iptables simulation

- **Real-time Dashboard**: Flask-based web interface with live metrics, training results, and ROC curves

- **Comprehensive Validation**: Phase 2 components fully tested and validated

---

## System Architecture

### High-Level Architecture

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

### Overall Workflow Sequence Diagram

```mermaid
sequenceDiagram
    participant PCAP as PCAP File
    participant Reader as PCAP Reader
    participant Window as Window Manager
    participant GPU as GPU Detector
    participant ML as ML Inference
    participant CUSUM as CUSUM/PCA
    participant Decision as Decision Engine
    participant Block as Blocking System
    participant Dashboard as Flask Dashboard

    PCAP->>Reader: Read packets
    Reader->>Window: Parse packets
    Window->>Window: Aggregate (1s/5s windows)
    Window->>GPU: Send window batches
    GPU->>GPU: Calculate entropy (OpenCL)
    GPU->>GPU: Extract features (OpenCL)
    GPU->>ML: Feature vectors
    GPU->>CUSUM: Entropy values
    ML->>ML: Random Forest inference
    CUSUM->>CUSUM: Statistical detection
    ML->>Decision: ML scores
    CUSUM->>Decision: Anomaly flags
    GPU->>Decision: Entropy scores
    Decision->>Decision: Ensemble decision
    Decision->>Block: Alert if attack detected
    Block->>Block: Update blackhole list
    Block->>Dashboard: Blocking metrics
    Decision->>Dashboard: Detection alerts
    GPU->>Dashboard: GPU utilization
    ML->>Dashboard: ML confidence scores
```

---

## Project Status

### ✅ Completed Phases

| Phase | Status | Description | Deliverables |
|-------|--------|-------------|--------------|
| **Phase 1** | ✅ Complete | Environment Setup & Dataset Acquisition | Setup scripts, datasets downloaded, GPU verified |
| **Phase 2** | ✅ Complete | Baseline CPU Implementation | PCAP reader, window manager, entropy/CUSUM/PCA detectors, comprehensive validation |
| **Phase 3** | ✅ Complete | ML Pipeline Training | Trained Random Forest model (**99.59% accuracy** with real entropy), training reports, ROC curves |
| **Phase 7** | ✅ Complete | Flask Dashboard | Real-time dashboard with metrics, alerts, training results |

### ⏳ Pending Phases

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 4** | ⏳ Pending | OpenCL GPU Implementation |
| **Phase 5** | ⏳ Pending | ML Inference Integration |
| **Phase 6** | ⏳ Pending | Blocking Implementation |
| **Phase 8** | ⏳ Pending | Performance Instrumentation |
| **Phase 9** | ⏳ Pending | Experiment Scripts & Evaluation |
| **Phase 10** | ⏳ Pending | Documentation & Report |

**Current Progress**: ~60% complete (Phases 1-3, 7 done)

---

## Requirements

- **Hardware**:
  - NVIDIA RTX 3060 GPU (or compatible OpenCL device)
  - 8GB+ RAM
  - 50GB+ disk space for datasets

- **Software**:
  - Ubuntu/Debian Linux
  - OpenCL drivers and libraries (`nvidia-opencl-icd`)
  - Python 3.9+
  - C++17 compiler (g++)
  - CMake 3.10+

---

## Installation

### 1. Install Dependencies

```bash
# Run setup script
./scripts/setup_environment.sh

# Or manually install:
sudo apt-get update
sudo apt-get install -y build-essential cmake libpcap-dev \
    ocl-icd-opencl-dev nvidia-opencl-icd \
    python3-numpy python3-pandas python3-matplotlib \
    python3-scipy python3-setuptools

pip3 install --user scikit-learn flask flask-socketio psutil joblib scapy pybind11
```

### 2. Verify GPU Setup

```bash
# Check GPU detection
nvidia-smi

# Verify OpenCL platforms
clinfo | grep "Platform Name"
```

### 3. Build the Project

```bash
mkdir -p build && cd build
cmake ..
make
```

### 4. Train ML Model

```bash
# Train Random Forest model
python3 src/ml/train_ml.py

# Model will be saved to models/rf_model.joblib
# Training report: reports/training_report.txt
# ROC curve: results/ml_roc_curve.png
```

---

## Usage

### Running Detection (After Phase 4-6 completion)

```bash
# Run detector on PCAP file
./build/detector --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap \
                 --window 1 --batch 128 --model models/rf_model.joblib
```

### Running Phase 2 Validation

```bash
# Run comprehensive Phase 2 tests
./tools/run_phase2_tests.sh

# Individual tests
python3 tools/validate_entropy.py      # Entropy validation
python3 tools/benchmark_phase2.py       # Performance benchmarks
python3 tools/test_pcap_processing.py  # PCAP processing test
```

### Starting Dashboard

```bash
# Start Flask dashboard
python3 src/dashboard/app.py

# Open browser to http://localhost:5000
# Dashboard shows:
# - Real-time GPU utilization
# - Throughput metrics
# - Detection alerts
# - ML training results (accuracy, ROC curve)
# - Blackhole list
```

### Running Experiments (After Phase 9 completion)

```bash
./scripts/run_experiment.sh --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap \
                            --pps 200000 --window 1 --batch 128
```

---

## Project Structure

```
project-root/
├── data/                          # Datasets
│   ├── cic-ddos2019/             # CIC-DDoS2019 dataset
│   ├── caida-ddos2007/           # CAIDA DDoS 2007 dataset
│   └── README.md                  # Dataset documentation
├── src/
│   ├── ingest/                    # Packet ingestion and windowing
│   │   ├── pcap_reader.cpp       # PCAP file reading (libpcap)
│   │   └── window_manager.cpp    # Tumbling windows (1s/5s)
│   ├── opencl/                    # GPU kernels and host code
│   │   ├── host.cpp              # OpenCL host setup
│   │   ├── gpu_detector.cpp      # GPU detector integration
│   │   └── kernels/
│   │       ├── entropy.cl        # GPU entropy kernel
│   │       └── feature_extract.cl # GPU feature extraction
│   ├── detectors/                 # Detection algorithms
│   │   ├── entropy_cpu.cpp       # CPU entropy baseline ✅
│   │   ├── cusum_detector.cpp    # CUSUM change detection ✅
│   │   ├── pca_detector.cpp      # PCA anomaly detection ✅
│   │   └── decision_engine.cpp   # Ensemble decision making
│   ├── ml/                        # Machine learning
│   │   ├── feature_extractor.py  # Feature extraction ✅
│   │   ├── train_ml.py           # Model training ✅
│   │   ├── inference_engine.cpp  # ML inference (Python C API)
│   │   ├── preprocessor.py       # Feature preprocessing ✅
│   │   └── features_spec.json    # Feature specification ✅
│   ├── blocking/                  # Blocking methods
│   │   ├── rtbh_controller.cpp   # RTBH controller
│   │   ├── pcap_filter.cpp       # PCAP filtering
│   │   └── iptables_simulator.cpp # iptables simulation
│   ├── dashboard/                 # Flask dashboard
│   │   ├── app.py                # Flask backend ✅
│   │   ├── templates/index.html  # Dashboard HTML ✅
│   │   └── static/js/dashboard.js # Frontend JavaScript ✅
│   └── utils/                     # Utilities
│       ├── logger.cpp             # Structured logging
│       ├── resource_monitor.cpp   # System monitoring
│       └── metrics_collector.cpp  # Metrics collection
├── scripts/
│   ├── setup_environment.sh      # Environment setup ✅
│   ├── prepare_data.py           # Dataset preprocessing ✅
│   ├── run_experiment.sh         # Experiment automation
│   ├── evaluate_detection.py     # Detection evaluation
│   └── plot_results.py           # Results plotting
├── tools/                         # Validation and testing
│   ├── validate_entropy.py       # Entropy validation ✅
│   ├── test_phase2.cpp           # C++ unit tests ✅
│   ├── test_pcap_processing.py   # PCAP processing test ✅
│   ├── benchmark_phase2.py       # Performance benchmarks ✅
│   └── run_phase2_tests.sh       # Test runner ✅
├── models/                        # Trained models
│   ├── rf_model.joblib          # Random Forest model ✅
│   └── preprocessor.joblib       # Feature preprocessor ✅
├── reports/                       # Training reports
│   ├── training_report.txt       # Detailed training report ✅
│   └── training_metrics.json     # Training metrics (JSON) ✅
├── results/                       # Plots and analysis
│   ├── ml_roc_curve.png         # ROC curve ✅
│   └── entropy_validation.png   # Entropy validation plots ✅
├── config/                        # Configuration files
│   └── detection_config.json     # Detection thresholds
├── logs/                          # Experiment logs (generated)
├── CMakeLists.txt                # Build configuration
├── README.md                      # This file
├── PHASE_STATUS.md               # Detailed phase status ✅
├── PHASE2_TESTING.md             # Phase 2 testing guide ✅
├── PRESENTATION_GUIDE.md         # Presentation guide ✅
├── MODEL_RETRAINING_SUMMARY.md   # Model improvement summary ✅
├── MODEL_IMPROVEMENT.md          # Model improvement guide ✅
└── plan.md                       # Implementation plan
```

---

## Implementation Phases

### Phase 1: Environment Setup ✅ COMPLETE

```mermaid
sequenceDiagram
    participant User
    participant Setup as Setup Script
    participant System as System Packages
    participant Python as Python Packages
    participant GPU as GPU Drivers
    participant Data as Datasets

    User->>Setup: Run setup_environment.sh
    Setup->>System: Install system packages
    System-->>Setup: Packages installed
    Setup->>Python: Install Python packages
    Python-->>Setup: Packages installed
    Setup->>GPU: Install NVIDIA OpenCL ICD
    GPU-->>Setup: GPU verified
    Setup->>Data: Download datasets
    Data-->>Setup: Datasets ready
    Setup-->>User: Environment ready
```

### Phase 2: Baseline CPU Implementation ✅ COMPLETE

```mermaid
sequenceDiagram
    participant PCAP as PCAP File
    participant Reader as PCAP Reader
    participant Window as Window Manager
    participant Entropy as Entropy CPU
    participant CUSUM as CUSUM Detector
    participant PCA as PCA Detector
    participant Validator as Validator

    PCAP->>Reader: Open PCAP file
    Reader->>Reader: Parse Ethernet/IP/TCP/UDP headers
    Reader->>Window: PacketInfo struct
    Window->>Window: Add to current window
    Window->>Window: Check window closure (1s/5s)
    Window->>Window: Build histograms (IPs, ports, sizes)
    Window->>Entropy: WindowStats
    Entropy->>Entropy: Calculate Shannon entropy
    Entropy->>Entropy: Compute entropy features
    Entropy->>CUSUM: Entropy values
    Entropy->>PCA: Entropy features
    CUSUM->>CUSUM: Update baseline & CUSUM statistic
    PCA->>PCA: Train/Detect anomalies
    CUSUM->>Validator: Detection results
    PCA->>Validator: Detection results
    Entropy->>Validator: Entropy values
    Validator->>Validator: Validate correctness
```

### Phase 3: ML Pipeline Training ✅ COMPLETE

```mermaid
sequenceDiagram
    participant CSV as CSV Files
    participant Loader as Data Loader
    participant Feature as Feature Extractor
    participant Entropy as Entropy Calculator
    participant Preproc as Preprocessor
    participant Model as Random Forest
    participant Eval as Evaluator
    participant Report as Report Generator

    CSV->>Loader: Load CSV files (16 files)
    Loader->>Loader: Label detection/inference
    Loader->>Feature: Combined DataFrame (800k rows)
    Feature->>Feature: Extract basic features
    Feature->>Entropy: Group into windows (1000 flows)
    Entropy->>Entropy: Calculate real entropy per window
    Entropy->>Entropy: Count unique IPs/ports
    Entropy->>Entropy: Calculate Top-N fractions
    Entropy->>Feature: Entropy features
    Feature->>Preproc: Feature matrix (24 features)
    Preproc->>Preproc: Remove outliers
    Preproc->>Model: Preprocessed features
    Model->>Model: Train Random Forest (200 trees, depth 15)
    Model->>Eval: Trained model
    Eval->>Eval: Test on hold-out set
    Eval->>Eval: Calculate metrics (accuracy, ROC AUC, F1)
    Eval->>Report: Metrics & feature importance
    Report->>Report: Generate training report
    Report->>Report: Save ROC curve plot
    Model->>Model: Save model (rf_model.joblib)
```

### Phase 4: OpenCL GPU Implementation ⏳ PENDING

```mermaid
sequenceDiagram
    participant CPU as CPU Host
    participant OpenCL as OpenCL Host
    participant GPU as GPU Device
    participant Kernel1 as Entropy Kernel
    participant Kernel2 as Feature Kernel
    participant Validator as GPU Validator

    CPU->>OpenCL: Initialize OpenCL context
    OpenCL->>GPU: Select NVIDIA RTX 3060
    OpenCL->>GPU: Create command queue
    CPU->>OpenCL: Transfer window batches (128-256)
    OpenCL->>GPU: Copy histograms to GPU memory
    OpenCL->>GPU: Launch entropy kernel
    GPU->>Kernel1: Parallel entropy calculation
    Kernel1->>Kernel1: Compute entropy (256 windows)
    Kernel1->>GPU: Entropy results
    OpenCL->>GPU: Launch feature extraction kernel
    GPU->>Kernel2: Parallel feature extraction
    Kernel2->>Kernel2: Build histograms, count features
    Kernel2->>GPU: Feature vectors
    GPU->>OpenCL: Copy results back
    OpenCL->>CPU: Feature vectors & entropy
    CPU->>Validator: Compare GPU vs CPU results
    Validator->>Validator: Validate correctness (<1e-5 diff)
```

### Detection Pipeline (Phase 5 Integration)

```mermaid
sequenceDiagram
    participant Window as Window Manager
    participant GPU as GPU Detector
    participant ML as ML Inference
    participant CUSUM as CUSUM Detector
    participant PCA as PCA Detector
    participant Decision as Decision Engine
    participant Block as Blocking System

    Window->>GPU: WindowStats batch
    GPU->>GPU: GPU entropy calculation
    GPU->>GPU: GPU feature extraction
    GPU->>ML: Feature vectors
    GPU->>CUSUM: Entropy values
    GPU->>PCA: Entropy features
    ML->>ML: Random Forest prediction
    CUSUM->>CUSUM: Update & detect change
    PCA->>PCA: Detect anomaly
    ML->>Decision: ML score (0-1)
    CUSUM->>Decision: CUSUM anomaly flag
    PCA->>Decision: PCA anomaly flag
    GPU->>Decision: Entropy score
    Decision->>Decision: Ensemble decision (OR logic)
    alt Attack Detected
        Decision->>Block: Alert with source IP
        Block->>Block: Add to blackhole list
        Block->>Block: Update RTBH rules
        Block->>Block: Update iptables rules
    else No Attack
        Decision->>Decision: Continue monitoring
    end
```

### Blocking Pipeline (Phase 6)

```mermaid
sequenceDiagram
    participant Decision as Decision Engine
    participant RTBH as RTBH Controller
    participant Filter as PCAP Filter
    participant IPTables as iptables Simulator
    participant Log as Logger
    participant Dashboard as Dashboard

    Decision->>RTBH: Alert (source IP, score)
    RTBH->>RTBH: Check if already blackholed
    alt Not Blackholed
        RTBH->>RTBH: Add IP to blackhole list
        RTBH->>Filter: Update filter rules
        RTBH->>IPTables: Generate iptables rule
        RTBH->>Log: Log blocking action
        RTBH->>Dashboard: Update blackhole list
        Filter->>Filter: Drop matching packets
        IPTables->>IPTables: Create rule file
    else Already Blackholed
        RTBH->>RTBH: Update timestamp
    end
    Filter->>Log: Packet drop statistics
    IPTables->>Log: Rule application metrics
    Log->>Dashboard: Blocking effectiveness
```

### Phase 1: Environment Setup ✅ COMPLETE

- Installed dependencies (OpenCL, libpcap, Python packages)
- Downloaded datasets (CIC-DDoS2019, CAIDA DDoS 2007)
- Verified GPU setup (NVIDIA RTX 3060)
- Created setup scripts

**Deliverables**: `scripts/setup_environment.sh`, `data/README.md`

### Phase 2: Baseline CPU Implementation ✅ COMPLETE

- PCAP packet ingestion (`pcap_reader.cpp`)
- Tumbling windows (`window_manager.cpp`)
- CPU entropy calculation (`entropy_cpu.cpp`)
- CUSUM change detection (`cusum_detector.cpp`)
- PCA anomaly detection (`pca_detector.cpp`)
- Comprehensive validation and testing

**Deliverables**: All Phase 2 C++ components, validation scripts, test suite

**Validation Results**:
- ✅ Entropy formula correctness: PASSED
- ✅ DDoS attack pattern detection: PASSED
- ✅ Unit tests: All components tested
- ✅ Performance: >100k packets/sec processing

### Phase 3: ML Pipeline Training ✅ COMPLETE

- Feature extraction from CSV data with **real entropy calculation**
- Random Forest model training (800k samples, 16 CSV files)
- Model evaluation and metrics
- Training reports and visualizations
- **Model retrained with real entropy features** (Phase 2 integration)

**Deliverables**: `models/rf_model.joblib`, `reports/training_report.txt`, `results/ml_roc_curve.png`

**Model Performance** (After Retraining with Real Entropy):
- Test Accuracy: **99.59%** ⬆️ (improved from 90.31%)
- ROC AUC: **0.9997** ⬆️ (improved from 0.9675)
- F1 Score: **0.9979** ⬆️ (improved from 0.9477)
- Precision: **99.97%**
- Recall: **99.61%** ⬆️ (improved from 90.28%)

**Key Features**:
- **Real entropy calculation** per window (Source IP, Destination IP, Ports, Protocol)
- **Real unique counts** (not constants)
- **Top-N fractions** calculated from actual data
- **24 features** extracted (including entropy features)
- **Feature importance**: Entropy features are most important (dst_ip_entropy: 16.01%)

### Phase 4: OpenCL GPU Implementation ⏳ PENDING

- OpenCL host setup
- GPU entropy kernel
- GPU feature extraction kernel
- GPU detector integration
- GPU correctness validation

### Phase 5: ML Inference Integration ⏳ PENDING

- Batched ML inference
- Decision engine integration
- Alert generation

### Phase 6: Blocking Implementation ⏳ PENDING

- RTBH controller and blocker
- iptables simulator
- Blocking effectiveness metrics

### Phase 7: Flask Dashboard ✅ COMPLETE

- Flask backend with REST API and WebSocket
- Real-time metrics visualization
- Training results display
- Interactive blocking controls

**Deliverables**: `src/dashboard/app.py`, templates, static files

---

## Testing & Validation

### Phase 2 Validation

Comprehensive test suite for Phase 2 components:

```bash
# Run all Phase 2 tests
./tools/run_phase2_tests.sh

# Individual tests
python3 tools/validate_entropy.py      # Entropy validation
python3 tools/benchmark_phase2.py       # Performance benchmarks
python3 tools/test_pcap_processing.py  # PCAP processing
```

**Test Coverage**:
- ✅ Entropy formula correctness
- ✅ DDoS attack pattern detection
- ✅ Unit tests for all components
- ✅ Integration tests
- ✅ Performance benchmarks

See `PHASE2_TESTING.md` for detailed testing guide.

### ML Model Evaluation

```bash
# View training report
cat reports/training_report.txt

# View training metrics
cat reports/training_metrics.json

# View ROC curve
xdg-open results/ml_roc_curve.png
```

---

## Configuration

Edit `config/detection_config.json` to adjust:
- Detection thresholds
- Ensemble weights
- Window sizes
- Batch sizes

Example configuration:
```json
{
  "entropy_threshold": 0.7,
  "ml_threshold": 0.5,
  "cusum_threshold": 5.0,
  "pca_threshold": 0.1,
  "ensemble_weights": {
    "entropy": 0.4,
    "ml": 0.4,
    "statistical": 0.2
  }
}
```

---

## Documentation

- **`PHASE_STATUS.md`**: Detailed status of all phases
- **`PHASE2_TESTING.md`**: Comprehensive testing guide for Phase 2
- **`PRESENTATION_GUIDE.md`**: Guide for presenting the project
- **`data/DATASET_ANALYSIS.md`**: Dataset information and structure
- **`plan.md`**: Complete implementation plan
- **`doc.md`**: Project requirements
- **`req.md`**: Implementation details

---

## Quick Start Guide

### For First-Time Users

1. **Install dependencies**:
   ```bash
   ./scripts/setup_environment.sh
   ```

2. **Build project**:
   ```bash
   mkdir -p build && cd build && cmake .. && make
   ```

3. **Train ML model**:
   ```bash
   python3 src/ml/train_ml.py
   ```

4. **Run validation**:
   ```bash
   python3 tools/validate_entropy.py
   ```

5. **Start dashboard**:
   ```bash
   python3 src/dashboard/app.py
   # Open http://localhost:5000
   ```

### For Developers

1. **Run Phase 2 tests**:
   ```bash
   ./tools/run_phase2_tests.sh
   ```

2. **Check project status**:
   ```bash
   cat PHASE_STATUS.md
   ```

3. **View training results**:
   ```bash
   cat reports/training_report.txt
   ```

---

## Key Metrics & Results

### ML Model Performance (After Retraining with Real Entropy)
- **Accuracy**: **99.59%** ⬆️ (improved from 90.31%)
- **ROC AUC**: **0.9997** ⬆️ (improved from 0.9675)
- **F1 Score**: **0.9979** ⬆️ (improved from 0.9477)
- **Precision**: **99.97%**
- **Recall**: **99.61%** ⬆️ (improved from 90.28%)
- **Top Features**: Entropy features dominate (dst_ip_entropy: 16.01%, protocol_entropy: 12.26%)

### Phase 2 Performance
- **Processing Speed**: >100k packets/sec
- **Entropy Calculation**: >1000 calc/sec
- **Memory Efficiency**: <100 bytes/packet

### Dataset Statistics
- **Training Samples**: 800,000
- **CSV Files**: 16 files
- **Test Samples**: 160,000
- **Training Samples**: 640,000

---

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA driver
nvidia-smi

# Install OpenCL ICD
sudo apt-get install nvidia-opencl-icd

# Verify OpenCL
clinfo | grep "Platform Name"
```

### Build Errors
```bash
# Install dependencies
sudo apt-get install libpcap-dev build-essential cmake

# Clean and rebuild
rm -rf build && mkdir build && cd build && cmake .. && make
```

### Python Import Errors
```bash
# Install Python packages
pip3 install --user scikit-learn flask flask-socketio psutil joblib

# Check Python path
python3 -c "import sys; print(sys.path)"
```

---

## License

Academic project - see project requirements for details.

---

## Contact & Support

For questions or issues:
1. Check `PHASE_STATUS.md` for current status
2. Review `PHASE2_TESTING.md` for testing issues
3. See `PRESENTATION_GUIDE.md` for presentation help
4. Check `plan.md` for implementation details

---

---

## Recent Updates

### ✅ Model Retraining with Real Entropy (November 2024)

- **Real entropy calculation** implemented using Phase 2 CPU baseline
- **Model retrained** with 24 features including real entropy values
- **Significant performance improvement**:
  - Accuracy: 90.31% → **99.59%** (+9.28%)
  - ROC AUC: 0.9675 → **0.9997** (+0.0322)
  - Feature importance: Entropy features now dominate (dst_ip_entropy: 16.01%)
- **10 features** now contribute meaningfully (vs 1 before)
- See `MODEL_RETRAINING_SUMMARY.md` for detailed comparison

### ✅ Phase 2 Validation Suite (November 2024)

- Comprehensive validation and testing framework created
- Entropy validation script (`tools/validate_entropy.py`)
- C++ unit tests (`tools/test_phase2.cpp`)
- Performance benchmarks (`tools/benchmark_phase2.py`)
- All Phase 2 components validated and tested

---

**Last Updated**: November 2024
**Current Version**: Phases 1-3 Complete (with real entropy), Phase 7 Complete
