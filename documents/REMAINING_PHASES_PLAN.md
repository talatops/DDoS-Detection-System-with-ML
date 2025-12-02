# Remaining Phases Implementation Plan

## Current Status Summary

### ✅ Completed Phases
- **Phase 1**: Environment Setup & Dataset Acquisition
- **Phase 2**: Baseline CPU Implementation (with comprehensive validation)
- **Phase 3**: ML Pipeline Training (99.59% accuracy with real entropy)
- **Phase 7**: Flask Dashboard (basic structure with training metrics)

### ⚠️ Partially Complete
- **Phase 4**: OpenCL GPU Implementation (kernels exist, host code exists, needs integration & testing)
- **Phase 5**: ML Inference Integration (inference engine exists, needs pipeline integration)
- **Phase 6**: Blocking Implementation (code exists, needs integration)
- **Phase 8**: Performance Instrumentation (utilities exist, needs integration)
- **Phase 9**: Experiment Scripts & Evaluation (scripts exist, need integration & testing)
- **Phase 10**: Documentation (mostly done, needs final report)

---

## Phase 4: Complete OpenCL GPU Implementation (Priority: HIGH)

### Current State
- ✅ OpenCL kernels exist (`entropy.cl`, `feature_extract.cl`)
- ✅ OpenCL host code exists (`host.cpp`, `host.h`)
- ✅ GPU detector skeleton exists (`gpu_detector.cpp`, `gpu_detector.h`)
- ❌ Kernels not fully integrated with main pipeline
- ❌ GPU correctness validation not implemented
- ❌ Batch processing not fully functional

### 4.1 Complete GPU Detector Implementation

**Tasks:**
1. **Complete `src/opencl/gpu_detector.cpp`**
   - Implement `prepareBatchData()` to convert `WindowStats` to GPU buffers
   - Implement `processBatch()` to:
     - Transfer data to GPU (async with `CL_FALSE`)
     - Launch entropy kernel for all features (src_ip, dst_ip, src_port, dst_port, packet_size, protocol)
     - Launch feature extraction kernels (histograms, unique counts)
     - Transfer results back (async)
     - Use event profiling to measure kernel execution time
   - Implement proper buffer management (create/destroy buffers)
   - Handle batch sizes of 128-256 windows

2. **Enhance `src/opencl/host.cpp`**
   - Add support for loading multiple kernels (`entropy.cl` and `feature_extract.cl`)
   - Implement async memory transfer functions
   - Add event profiling helpers
   - Add pinned memory allocation (if supported)

3. **Update `src/opencl/kernels/entropy.cl`**
   - Verify kernel works with multi-feature entropy (`compute_multi_entropy`)
   - Test with RTX 3060 work-group sizes (32, 64, 128, 256)

4. **Update `src/opencl/kernels/feature_extract.cl`**
   - Verify histogram kernels work correctly
   - Optimize for RTX 3060 memory hierarchy

**Files to Modify:**
- `src/opencl/gpu_detector.cpp` (complete implementation)
- `src/opencl/gpu_detector.h` (add missing methods if needed)
- `src/opencl/host.cpp` (add multi-kernel support)
- `src/opencl/host.h` (add missing declarations)

**Deliverables:**
- Fully functional GPU detector that processes batches of windows
- GPU entropy calculation for all 6 features
- GPU feature extraction (histograms, unique counts, packet stats)

### 4.2 GPU Correctness Validation

**Tasks:**
1. **Create `tools/validate_gpu_correctness.py`**
   - Load test windows from CSV or generate synthetic data
   - Run CPU entropy calculation (using `entropy_cpu.cpp`)
   - Run GPU entropy calculation (using `gpu_detector.cpp`)
   - Compare results with tolerance 1e-5
   - Generate comparison report
   - Plot GPU vs CPU entropy values

2. **Create `tools/benchmark_gpu_performance.cpp`**
   - Benchmark CPU entropy calculation time
   - Benchmark GPU entropy calculation time (including transfers)
   - Measure speedup for different batch sizes (32, 64, 128, 256, 512)
   - Measure GPU utilization using NVML
   - Generate performance report

**Files to Create:**
- `tools/validate_gpu_correctness.py`
- `tools/benchmark_gpu_performance.cpp`
- `tools/run_gpu_tests.sh` (test runner script)

**Deliverables:**
- Validation script showing GPU correctness (within tolerance)
- Performance benchmarks showing GPU speedup
- GPU utilization metrics

### 4.3 Integration Testing

**Tasks:**
1. **Update `src/main.cpp`**
   - Integrate GPU detector into main pipeline
   - Replace CPU entropy with GPU entropy when GPU is available
   - Add fallback to CPU if GPU initialization fails
   - Add command-line flags: `--use-gpu` (default), `--use-cpu`, `--dashboard-url <url>` (optional)
   - If dashboard URL provided, initialize HTTP client or WebSocket connection for live updates

2. **Test Integration**
   - Run with small PCAP file (1000 packets)
   - Verify GPU processes windows correctly
   - Verify results match CPU baseline
   - Measure end-to-end latency

**Files to Modify:**
- `src/main.cpp` (integrate GPU detector)

**Deliverables:**
- Main pipeline using GPU for entropy and feature extraction
- Fallback mechanism to CPU if GPU unavailable

---

## Phase 5: Complete ML Inference Integration (Priority: HIGH)

### Current State
- ✅ ML inference engine exists (`inference_engine.cpp`, `inference_engine.h`)
- ✅ Preprocessor loading implemented
- ✅ Decision engine exists (`decision_engine.cpp`, `decision_engine.h`)
- ❌ Not integrated into main pipeline
- ❌ Feature extraction from GPU results not implemented
- ❌ Batch inference not fully tested

### 5.1 Complete Feature Extraction Pipeline

**Tasks:**
1. **Create `src/ml/feature_builder.cpp`**
   - Convert GPU entropy results to ML feature vector
   - Combine GPU features with window statistics
   - Build feature vector matching training features:
     - Entropy features (6): src_ip, dst_ip, src_port, dst_port, packet_size, protocol
     - Window stats: packet_count, byte_count, flow_count
     - Unique counts: unique_src_ips, unique_dst_ips, unique_ports
     - Top-N fractions: top10_src_ip_fraction, top10_dst_ip_fraction
     - Packet stats: avg_packet_size, std_packet_size

2. **Update `src/ml/inference_engine.cpp`**
   - Verify batch inference works correctly
   - Add error handling for Python API calls
   - Add logging for inference times

**Files to Create:**
- `src/ml/feature_builder.cpp`
- `src/ml/feature_builder.h`

**Files to Modify:**
- `src/ml/inference_engine.cpp` (add batch processing verification)

**Deliverables:**
- Feature builder that converts GPU results to ML features
- Batch inference working correctly

### 5.2 Integrate ML Inference into Main Pipeline

**Tasks:**
1. **Update `src/main.cpp`**
   - Initialize ML inference engine
   - Load model and preprocessor from `models/`
   - After GPU processing, extract ML features
   - Run ML inference on batch of windows
   - Pass ML probabilities to decision engine
   - When attack detected, send alert to dashboard via WebSocket or HTTP POST (if dashboard URL provided)

2. **Update `src/detectors/decision_engine.cpp`**
   - Verify decision engine receives ML probabilities correctly
   - Test ensemble detection (entropy OR ML OR CUSUM/PCA)
   - Test weighted detection (weighted combination)

3. **Create `config/detection_config.json`**
   ```json
   {
     "entropy_threshold": 0.7,
     "ml_threshold": 0.5,
     "cusum_threshold": 3.0,
     "pca_threshold": 2.0,
     "use_weighted": false,
     "weights": {
       "entropy": 0.4,
       "ml": 0.4,
       "cusum": 0.1,
       "pca": 0.1
     }
   }
   ```

**Files to Modify:**
- `src/main.cpp` (integrate ML inference, send alerts to dashboard)
- `src/detectors/decision_engine.cpp` (verify integration)
- `src/dashboard/app.py` (add `/api/alerts` POST endpoint to receive real-time alerts, update WebSocket broadcasts)

**Files to Create:**
- `config/detection_config.json`
- `src/utils/config_loader.cpp` (optional, for loading config)
- `src/utils/dashboard_client.cpp` (HTTP client for sending alerts to dashboard, optional - or use libcurl)
- `src/utils/dashboard_client.h` (Dashboard client interface)

**Deliverables:**
- ML inference integrated into main pipeline
- Decision engine combining all detectors
- Configurable thresholds and weights

### 5.3 End-to-End ML Testing

**Tasks:**
1. **Create `tools/test_ml_integration.py`**
   - Load test windows from labeled CSV
   - Run through GPU → ML → Decision pipeline
   - Compare predictions with ground truth
   - Compute accuracy, precision, recall, F1

2. **Test with Real Data**
   - Run on CIC-DDoS2019 PCAP files
   - Verify ML detects attacks correctly
   - Measure false positive rate

**Files to Create:**
- `tools/test_ml_integration.py`

**Deliverables:**
- ML integration test showing correct predictions
- Performance metrics on test dataset

---

## Phase 6: Complete Blocking Implementation (Priority: MEDIUM)

### Current State
- ✅ RTBH controller exists (`rtbh_controller.cpp`, `rtbh_controller.h`)
- ✅ PCAP filter exists (`pcap_filter.cpp`, `pcap_filter.h`)
- ✅ iptables simulator exists (`iptables_simulator.cpp`, `iptables_simulator.h`)
- ❌ Not integrated into main pipeline
- ❌ Blocking metrics not implemented
- ❌ Blocking effectiveness not measured

### 6.1 Integrate Blocking into Main Pipeline

**Tasks:**
1. **Update `src/main.cpp`**
   - Initialize RTBH controller and iptables simulator
   - Add `--enable-iptables` command-line flag (requires explicit enable)
   - When decision engine detects attack:
     - Extract source IPs from attack windows
     - Add IPs to RTBH blackhole list
     - Generate iptables rules
     - If `--enable-iptables` flag is set AND running with sudo:
       - Execute iptables rules using `IptablesSimulator::executeRules(false)` (dry_run=false)
       - Add safety checks: verify sudo privileges, confirm before execution, limit number of rules
     - Log blocking actions with timestamp
   - Apply PCAP filter during packet replay (if using tcpreplay)

2. **Implement Blocking Metrics**
   - Track: packets dropped, bytes dropped, IPs blackholed
   - Track: false positives (legitimate IPs blocked)
   - Track: blocking latency (alert → rule application time)

3. **Create `src/blocking/blocking_metrics.cpp`**
   - Collect blocking statistics
   - Write to `logs/blocking.csv`
   - Format: `timestamp, ip, action, packets_dropped, bytes_dropped, is_false_positive`

**Files to Modify:**
- `src/main.cpp` (integrate blocking, add --enable-iptables flag, send blocking updates to dashboard)
- `src/blocking/rtbh_controller.cpp` (add logging)
- `src/blocking/iptables_simulator.cpp` (add logging, enable execution with explicit flag and safety checks - check sudo, confirm before execution)
- `src/dashboard/app.py` (add `/api/blocking` endpoint to receive blocking updates, update blackhole list display in real-time)

**Files to Create:**
- `src/blocking/blocking_metrics.cpp`
- `src/blocking/blocking_metrics.h`

**Deliverables:**
- Blocking integrated into main pipeline
- Blocking metrics collection
- Logging of blocking actions

### 6.2 Blocking Effectiveness Testing

**Tasks:**
1. **Create `tools/test_blocking.py`**
   - Load attack PCAP file
   - Run detection and blocking
   - Measure: % attack packets dropped, % legitimate packets dropped
   - Generate blocking effectiveness report

2. **Test RTBH and iptables**
   - Verify RTBH blackhole list updates correctly
   - Verify iptables rules generated correctly
   - Test with multiple attack IPs

**Files to Create:**
- `tools/test_blocking.py`

**Deliverables:**
- Blocking effectiveness test
- Metrics showing attack packet drop rate

---

## Phase 8: Complete Performance Instrumentation (Priority: MEDIUM)

### Current State
- ✅ Logger exists (`logger.cpp`, `logger.h`)
- ✅ Metrics collector exists (`metrics_collector.cpp`, `metrics_collector.h`)
- ✅ Resource monitor exists (`resource_monitor.cpp`, `resource_monitor.h`)
- ❌ Not fully integrated into main pipeline
- ❌ GPU metrics not collected
- ❌ Detailed timing not implemented

### 8.1 Complete Timing Instrumentation

**Tasks:**
1. **Add Timing Throughout Pipeline**
   - Ingestion time (PCAP reading)
   - Windowing time (window creation)
   - GPU transfer time (CPU → GPU)
   - GPU kernel execution time (from OpenCL events)
   - GPU transfer time (GPU → CPU)
   - ML inference time
   - Decision engine time
   - Blocking time
   - Total latency per window

2. **Update `src/utils/metrics_collector.cpp`**
   - Add timing collection for each stage
   - Write to `logs/timing.csv`
   - Format: `timestamp, stage, time_ms, window_id`

**Files to Modify:**
- `src/main.cpp` (add timing instrumentation)
- `src/utils/metrics_collector.cpp` (add timing collection)
- `src/opencl/gpu_detector.cpp` (add timing from OpenCL events)

**Deliverables:**
- Complete timing instrumentation
- Timing logs for all pipeline stages

### 8.2 Complete Resource Monitoring

**Tasks:**
1. **Update `src/utils/resource_monitor.cpp`**
   - Add GPU utilization monitoring (using NVML)
   - Add GPU memory usage monitoring
   - Add per-core CPU usage
   - Update frequency: 1 second

2. **Integrate into Main Pipeline**
   - Start resource monitor thread
   - Collect metrics every second
   - Write to `logs/metrics.csv`

**Files to Modify:**
- `src/utils/resource_monitor.cpp` (add GPU monitoring)
- `src/main.cpp` (start resource monitor thread)

**Deliverables:**
- Complete resource monitoring (CPU, GPU, memory)
- Resource metrics logs

### 8.3 Enhanced Logging

**Tasks:**
1. **Update `src/utils/logger.cpp`**
   - Add structured logging for alerts
   - Format: `logs/alerts.csv`: `timestamp, window_start, src_ip, score, detector_type, is_attack`
   - Format: `logs/kernel_times.csv`: `timestamp, kernel_name, execution_time_ms, batch_size`

2. **Add Log Rotation**
   - Rotate logs when they exceed size limit
   - Keep last N log files

**Files to Modify:**
- `src/utils/logger.cpp` (add alert logging, kernel timing)

**Deliverables:**
- Enhanced logging system
- Structured logs for all events

---

## Phase 9: Complete Experiment Scripts & Evaluation (Priority: MEDIUM)

### Current State
- ✅ Experiment scripts exist (`run_experiment.sh`, `evaluate_detection.py`, `plot_results.py`)
- ❌ Scripts not fully integrated
- ❌ Evaluation metrics not computed
- ❌ Plots not generated

### 9.1 Complete Experiment Automation

**Tasks:**
1. **Update `scripts/run_experiment.sh`**
   - Start Flask dashboard FIRST: `python3 src/dashboard/app.py &` (for live updates during experiments)
   - Start metrics collector in background
   - Start resource monitor thread (handled by main.cpp)
   - Start detector with PCAP file: `./build/detector --pcap <file> --window <1|5> --batch <size> --use-gpu --dashboard-url http://localhost:5000`
   - Detector sends real-time updates to dashboard via WebSocket or HTTP POST (if dashboard URL provided)
   - Dashboard polls `logs/alerts.csv` every 1 second for new alerts (alternative: C++ detector sends alerts via HTTP POST to `/api/alerts` endpoint)
   - Dashboard displays real-time: detection alerts, GPU utilization, throughput, blackhole list updates
   - Wait for detector completion
   - Collect logs after completion
   - Generate plots automatically

2. **Add Experiment Configuration**
   - Support command-line arguments:
     - `--pcap <file>`: PCAP file to process
     - `--pps <rate>`: Packet rate (for tcpreplay)
     - `--window <secs>`: Window size (1 or 5)
     - `--batch <size>`: Batch size (128, 256, 512)
     - `--use-gpu`: Use GPU (default) or CPU
     - `--model <path>`: ML model path
     - `--config <path>`: Detection config path
     - `--enable-iptables`: Enable iptables rule execution (requires sudo)
     - `--dashboard-url <url>`: Dashboard URL for live updates (default: http://localhost:5000)

**Files to Modify:**
- `scripts/run_experiment.sh` (complete implementation, start dashboard first for live updates)
- `src/dashboard/app.py` (add real-time WebSocket updates, HTTP POST endpoints for alerts/blocking)
- `src/main.cpp` (add dashboard communication for live detection results)

**Deliverables:**
- Fully automated experiment script
- Support for all configuration options

### 9.2 Complete Evaluation Scripts

**Tasks:**
1. **Update `scripts/evaluate_detection.py`**
   - Load alerts from `logs/alerts.csv`
   - Load ground truth from labeled dataset
   - Compute metrics:
     - Precision, Recall, F1, FPR, TPR
     - Detection lead time (attack start → first alert)
     - Per-attack-type analysis
   - Generate evaluation report

2. **Add Blocking Evaluation**
   - Load blocking logs
   - Compute blocking effectiveness
   - Compute false positive rate (legitimate IPs blocked)

**Files to Modify:**
- `scripts/evaluate_detection.py` (complete implementation)

**Deliverables:**
- Complete evaluation script
- Evaluation metrics report

### 9.3 Complete Plotting Scripts

**Tasks:**
1. **Update `scripts/plot_results.py`**
   - Generate all required plots:
     - Detection lead time vs packet rate
     - Throughput vs batch size
     - TPR/FPR curves
     - Blocking effectiveness
     - GPU utilization timeline
     - Latency distribution (avg, 95th percentile)
     - Scalability analysis
   - Save plots to `results/` directory

**Files to Modify:**
- `scripts/plot_results.py` (complete implementation)

**Deliverables:**
- Complete plotting script
- All required plots generated

### 9.4 Run Comprehensive Experiments

**Tasks:**
1. **Correctness Experiments**
   - GPU vs CPU entropy comparison
   - ML prediction accuracy on test set

2. **Performance Experiments**
   - Throughput: Vary rates (100k, 200k, 500k, 1M pps)
   - Latency: Per-window processing time
   - Scalability: Batch size impact

3. **Accuracy Experiments**
   - Detection metrics on labeled dataset
   - Per-attack-type analysis

4. **Blocking Experiments**
   - Blocking effectiveness
   - False positive rate

5. **Ablation Studies**
   - Entropy-only vs ML-only vs Combined vs All-three

**Deliverables:**
- Comprehensive experiment results
- All plots and metrics

---

## Phase 10: Final Documentation & Report (Priority: LOW)

### Current State
- ✅ README.md exists (comprehensive)
- ✅ Phase documentation exists
- ✅ Testing guides exist
- ❌ Technical report not written
- ❌ Presentation slides not created

### 10.1 Technical Report (10-12 pages)

**Tasks:**
1. **Write Technical Report**
   - Abstract & CCP justification
   - System architecture (with diagrams)
   - Detection algorithms (entropy math, RF hyperparameters, CUSUM/PCA)
   - OpenCL kernel design (with code snippets)
   - RTBH + iptables architecture
   - Experimental setup (hardware: RTX 3060 specs, software versions)
   - Results (all required metrics with tables/plots)
   - Discussion (trade-offs, limitations)
   - Conclusion & future work

**Files to Create:**
- `report/technical_report.md` or `report/technical_report.pdf`

**Deliverables:**
- Complete technical report (10-12 pages)

### 10.2 Presentation Slides (10-12 slides)

**Tasks:**
1. **Create Presentation Slides**
   - System overview
   - Architecture diagram
   - Detection algorithms
   - GPU acceleration highlights
   - Key results (metrics, plots)
   - Demo screenshots
   - Conclusion

**Files to Create:**
- `report/presentation.pdf` or `report/presentation.pptx`

**Deliverables:**
- Presentation slides (10-12 slides)

### 10.3 Final Code Documentation

**Tasks:**
1. **Add Doxygen Comments**
   - Document all C++ classes and functions
   - Generate Doxygen HTML documentation

2. **Add Python Docstrings**
   - Document all Python functions
   - Generate Sphinx documentation (optional)

**Files to Modify:**
- All C++ source files (add Doxygen comments)
- All Python scripts (add docstrings)

**Deliverables:**
- Complete code documentation

---

## Implementation Priority & Timeline

### Week 1: Phase 4 (GPU Implementation)
- **Days 1-3**: Complete GPU detector implementation
- **Days 4-5**: GPU correctness validation
- **Days 6-7**: Integration testing

### Week 2: Phase 5 (ML Integration)
- **Days 1-2**: Complete feature extraction pipeline
- **Days 3-4**: Integrate ML inference into main pipeline
- **Days 5-7**: End-to-end ML testing

### Week 3: Phase 6 & 8 (Blocking & Instrumentation)
- **Days 1-3**: Complete blocking implementation
- **Days 4-5**: Complete performance instrumentation
- **Days 6-7**: Integration testing

### Week 4: Phase 9 & 10 (Experiments & Documentation)
- **Days 1-3**: Complete experiment scripts
- **Days 4-5**: Run comprehensive experiments
- **Days 6-7**: Write technical report and presentation

**Total: ~4 weeks of focused development**

---

## Key Questions Before Starting

1. **GPU Testing**: Do you have test PCAP files ready, or should we generate synthetic test data?
2. **ML Model**: Is the current model (`models/rf_model.joblib`) the final version, or do you want to retrain after GPU integration?
3. **Blocking Execution**: Should iptables rules be executed (requires sudo) or just simulated?
4. **Dashboard Integration**: Should the dashboard show live detection results during experiments, or only post-experiment analysis?
5. **Experiment Priority**: Which experiments are most important for your presentation/report?

---

## Success Criteria

1. ✅ GPU processes 200k+ pps on RTX 3060
2. ✅ GPU correctness validated (within tolerance)
3. ✅ ML inference integrated and working
4. ✅ All three detectors (Entropy, ML, CUSUM/PCA) working together
5. ✅ Blocking (RTBH + iptables) functional
6. ✅ All required metrics measured and logged
7. ✅ Detection lead time < 5 seconds for attacks
8. ✅ Accuracy metrics: F1 > 0.85, FPR < 0.05
9. ✅ Comprehensive experiment results
10. ✅ Technical report and presentation complete

