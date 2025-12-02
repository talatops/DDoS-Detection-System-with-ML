# GPU Usage in DDoS Detection System

## Current GPU Usage Strategy

### ✅ GPU-Accelerated Components (OpenCL)

1. **Entropy Calculation** (Heavy GPU usage)
   - **Location**: `src/opencl/kernels/entropy.cl`
   - **When**: During runtime detection (processing PCAP files)
   - **What**: Parallel entropy calculation for multiple windows simultaneously
   - **GPU Load**: HIGH (1000s of parallel calculations)

2. **Feature Extraction** (Heavy GPU usage)
   - **Location**: `src/opencl/kernels/feature_extract.cl`
   - **When**: During runtime detection
   - **What**: Histogram accumulation, counting, aggregation
   - **GPU Load**: HIGH (parallel processing of packet batches)

3. **Batch Processing** (Moderate GPU usage)
   - **Location**: `src/opencl/gpu_detector.cpp`
   - **When**: Processing 128-256 windows in parallel
   - **What**: Batched entropy and feature extraction
   - **GPU Load**: MODERATE-HIGH (depends on batch size)

### ❌ CPU-Based Components

1. **ML Training** (CPU only)
   - **Location**: `src/ml/train_ml.py`
   - **Why**: scikit-learn Random Forest runs on CPU
   - **Note**: This is fine - training happens once, inference is fast

2. **ML Inference** (CPU only)
   - **Location**: `src/ml/inference_engine.cpp`
   - **Why**: Python C API calls scikit-learn (CPU-based)
   - **Note**: Inference is fast enough on CPU for real-time

3. **CUSUM/PCA Detection** (CPU only)
   - **Location**: `src/detectors/cusum_detector.cpp`, `pca_detector.cpp`
   - **Why**: Statistical algorithms, not computationally intensive
   - **Note**: These are fast on CPU

## Maximizing GPU Usage

### During Runtime Detection (Main GPU Usage)

The GPU will be **heavily utilized** during packet processing:

```
PCAP File → Packets
    ↓
Window Manager (CPU) → Creates windows
    ↓
GPU Batch Processing (128-256 windows)
    ├── Entropy Calculation (GPU) ← HEAVY GPU LOAD
    ├── Feature Extraction (GPU) ← HEAVY GPU LOAD
    └── Histogram Building (GPU) ← HEAVY GPU LOAD
    ↓
CPU: ML Inference (fast, uses GPU features)
CPU: CUSUM/PCA (fast)
    ↓
Decision Engine → Alerts
```

### GPU Utilization Metrics

Expected GPU usage during detection:
- **Idle**: 0-5%
- **Processing PCAP**: 60-95% (depends on packet rate)
- **Peak**: 95-100% during batch processing

### How to Maximize GPU Usage

1. **Increase Batch Size**
   ```cpp
   // In main.cpp or config
   size_t batch_size = 256;  // Larger = more GPU work
   ```

2. **Process Multiple Windows in Parallel**
   ```cpp
   // Process 1s and 5s windows simultaneously
   gpu_detector.computeEntropyBatch(windows_1s);
   gpu_detector.computeEntropyBatch(windows_5s);
   ```

3. **Use Async Processing**
   ```cpp
   // Overlap GPU computation with CPU work
   cl::Event gpu_event;
   gpu_detector.computeEntropyAsync(windows, &gpu_event);
   // Do CPU work while GPU computes
   // Wait for GPU when needed
   gpu_event.wait();
   ```

4. **Process High Packet Rates**
   ```bash
   # Higher PPS = more GPU work
   tcpreplay --pps=500000 data/test.pcap
   ```

## GPU Usage During Training

### Current: CPU-Based Training

The ML training script (`train_ml.py`) uses CPU because:
- scikit-learn Random Forest is CPU-based
- Training happens once (not real-time)
- CPU is sufficient for training

### Optional: GPU-Accelerated Feature Extraction During Training

If you want to use GPU during training (for feature extraction):

1. **Preprocess PCAP files with GPU**
   ```python
   # In train_ml.py, add GPU feature extraction
   from src.opencl.gpu_detector import GpuDetector
   
   gpu = GpuDetector()
   gpu.initialize()
   
   # Extract features using GPU
   for window in windows:
       entropy = gpu.computeEntropy(window)
   ```

2. **Use GPU for Entropy Calculation**
   - Calculate entropy for training data using GPU
   - This maximizes GPU usage during data preparation

## Monitoring GPU Usage

### During Detection

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: Run detector
./bin/detector --pcap data/test.pcap --window 1 --batch 256
```

### Expected Output

```
GPU Utilization: 85-95%
Memory Usage: 200-500 MB
Power: 80-120W (RTX 3060)
```

## Summary

### ✅ GPU is Maximized During:
- **Runtime Detection** (main use case)
  - Entropy calculation: 1000s of parallel operations
  - Feature extraction: Parallel histogram building
  - Batch processing: 128-256 windows simultaneously

### ❌ GPU Not Used During:
- **ML Training** (CPU-based, happens once)
- **ML Inference** (CPU-based, fast enough)
- **CUSUM/PCA** (CPU-based, not intensive)

### 🎯 To Maximize GPU Usage:
1. Use large batch sizes (256+)
2. Process high packet rates (200K+ PPS)
3. Use async GPU operations
4. Process multiple window sizes simultaneously

**Bottom Line**: GPU will be **heavily utilized** (80-95%) during runtime detection, which is the main workload. Training is CPU-based but happens only once.

