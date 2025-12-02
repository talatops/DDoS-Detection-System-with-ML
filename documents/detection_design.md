## Detection and Classification Design

This document explains how the system classifies windows of network traffic as **ATTACK** vs **benign** using entropy-based statistics, CUSUM and PCA change detectors, and machine‑learning models. It follows the data from raw packets all the way to alerts and optional RTBH-style blocking, and points to the concrete implementation files in the repository.

---

## 1. End‑to‑end pipeline overview

At a high level, detection is done **per time window**, not per individual packet:

- **Packet ingestion and parsing** (`src/ingest/pcap_reader.*`)
  - Reads packets from a PCAP file, parses IP/TCP/UDP headers into a `PacketInfo` struct with timestamps, 5‑tuple, protocol, size, and TCP flags.
- **Window aggregation** (`src/ingest/window_manager.*`)
  - Groups packets into fixed‑length time windows (default **1 second**), maintaining histograms and summary statistics (`WindowStats`) per window.
- **Entropy and feature computation**
  - **GPU entropy path**: `GPUDetector` (`src/opencl/gpu_detector.cpp`) + OpenCL kernels compute six entropy values per window in parallel.
  - **CPU entropy path**: `EntropyDetector` (`src/detectors/entropy_cpu.*`) computes the same entropies if GPU is unavailable.
  - **ML features**: `FeatureBuilder` (`src/ml/feature_builder.cpp`) derives a 24‑dimensional feature vector from each `WindowStats` + entropy values.
- **Machine‑learning inference**
  - `MLInferenceEngine` (`src/ml/inference_engine.*`) loads a scikit‑learn model (`*.joblib`) and an optional preprocessor to produce **attack probabilities** for each window.
- **Ensemble decision**
  - `DecisionEngine` (`src/detectors/decision_engine.*`) combines entropy anomaly score, ML probability, CUSUM change statistic, and PCA reconstruction error into a **per‑window classification**: ATTACK vs benign.
- **Alerting and blocking**
  - Runtime glue is in `src/main.cpp`: it batches windows, runs GPU/CPU + ML + `DecisionEngine`, logs alerts, and passes top source IPs to the RTBH controller (`src/blocking/rtbh_controller.*`) and filter (`src/blocking/pcap_filter.*`) to drop future packets from attackers.

From a detection perspective, **any window whose `DetectionResult::is_attack` is true is treated as ATTACK**; windows that do not cross the ensemble/weighted thresholds are implicitly benign (no alert, no blocking).

---

## 2. Windowing and data aggregation

### 2.1 Packet ingestion

Packets are read and parsed by `PcapReader`:

```22:52:src/ingest/pcap_reader.h
struct PacketInfo {
    uint64_t timestamp_us;    // Microsecond timestamp
    uint32_t src_ip;
    uint32_t dst_ip;
    uint16_t src_port;
    uint16_t dst_port;
    uint8_t  protocol;        // 6=TCP, 17=UDP, etc.
    uint16_t packet_size;     // IP packet length in bytes
    uint8_t  tcp_flags;       // SYN, FIN, RST, ACK bits
    bool     is_valid;
};
```

The implementation in `pcap_reader.cpp` parses different link types (raw IP vs Ethernet), extracts IP and transport headers, and populates `PacketInfo`. Only valid IPv4 packets are passed onward.

### 2.2 Time windows and `WindowStats`

`WindowManager` constructs fixed‑size time windows:

```66:72:src/ingest/window_manager.h
class WindowManager {
public:
    using WindowCallback = std::function<void(const WindowStats&)>;
    
    WindowManager(uint32_t window_size_sec = 1);
    ...
private:
    uint32_t window_size_us_;  // Window size in microseconds
    WindowStats current_window_;
    uint64_t window_start_time_us_;
    WindowCallback callback_;
};
```

The **window size** is controlled by:

- Command line argument `--window <seconds>` in `main.cpp` (`window_size` defaults to **1 second**).
- That value is passed to `WindowManager(window_size)` and converted to microseconds as `window_size_us_ = window_size_sec * 1e6`.

Per window, `WindowStats` accumulates histograms and counts:

```10:42:src/ingest/window_manager.h
struct WindowStats {
    uint64_t window_start_us;
    uint64_t window_end_us;
    
    // Histograms for entropy calculation
    std::unordered_map<uint32_t, uint32_t> src_ip_counts;
    std::unordered_map<uint32_t, uint32_t> dst_ip_counts;
    std::unordered_map<uint16_t, uint32_t> src_port_counts;
    std::unordered_map<uint16_t, uint32_t> dst_port_counts;
    std::unordered_map<uint16_t, uint32_t> packet_size_counts;
    std::unordered_map<uint8_t,  uint32_t> protocol_counts;
    
    // Aggregated statistics
    uint32_t total_packets;
    uint64_t total_bytes;
    uint32_t unique_src_ips;
    uint32_t unique_dst_ips;
    uint32_t flow_count;
    uint32_t tcp_packets;
    uint32_t udp_packets;
    uint32_t syn_packets;
    uint32_t fin_packets;
    uint32_t rst_packets;
    uint32_t ack_packets;
    
    // Flow tracking (5‑tuple)
    std::unordered_map<uint64_t, uint32_t> flow_counts;
    ...
};
```

Each incoming `PacketInfo` updates the current window:

```28:52:src/ingest/window_manager.cpp
void WindowManager::addPacket(const PacketInfo& packet) {
    if (!packet.is_valid) return;
    
    if (window_start_time_us_ == 0) {
        window_start_time_us_ = packet.timestamp_us;
        current_window_.window_start_us = packet.timestamp_us;
    }
    current_window_.window_end_us = packet.timestamp_us;
    
    // Histograms
    current_window_.src_ip_counts[packet.src_ip]++;
    current_window_.dst_ip_counts[packet.dst_ip]++;
    current_window_.src_port_counts[packet.src_port]++;
    current_window_.dst_port_counts[packet.dst_port]++;
    current_window_.packet_size_counts[packet.packet_size]++;
    current_window_.protocol_counts[packet.protocol]++;
    
    // Aggregates
    current_window_.total_packets++;
    current_window_.total_bytes += packet.packet_size;
    ...
}
```

The window is closed when the elapsed time reaches `window_size_us_`:

```77:85:src/ingest/window_manager.cpp
void WindowManager::checkWindow(uint64_t current_time_us) {
    if (window_start_time_us_ == 0) return;
    uint64_t elapsed = current_time_us - window_start_time_us_;
    if (elapsed >= window_size_us_) {
        closeWindow();
    }
}
```

On `closeWindow`, the final `[start, end)` times are set and the `WindowStats` are passed to the registered callback, where they are batched for detection (`window_batch` in `main.cpp`).

**Interpretation:** detection always decides **per window of length `--window` seconds**, using all packets that arrived in that time slice.

---

## 3. Entropy‑based features and intuition

### 3.1 Shannon entropy reminder

For a discrete distribution \( p_i \) over symbols \( i \), **Shannon entropy** is:

\[
H = -\sum_i p_i \log_2 p_i
\]

This measures how “spread out” or “unpredictable” the distribution is. In DDoS attacks, we typically see:

- **Low source IP entropy**: many packets from the same or few attack sources (or a small botnet).
- **Low destination IP entropy**: traffic focused on a single victim or small set of victims.
- **Low port entropy**: same service/port being hammered.

Thus, **lower entropy ⇒ more “attack‑like”**.

### 3.2 Entropy computation in code

`EntropyDetector` wraps the entropy calculation for the histograms in `WindowStats`:

```11:24:src/detectors/entropy_cpu.cpp
double EntropyDetector::calculateEntropy(const std::unordered_map<uint32_t, uint32_t>& counts, uint32_t total) {
    if (total == 0) return 0.0;
    
    double entropy = 0.0;
    const double log2 = std::log(2.0);
    
    for (const auto& pair : counts) {
        if (pair.second > 0) {
            double p = static_cast<double>(pair.second) / static_cast<double>(total);
            entropy -= p * std::log(p) / log2;  // log2
        }
    }
    return entropy;
}
```

The same pattern is used for 16‑bit (`uint16_t`) and 8‑bit (`uint8_t`) histograms. Entropy features for a window are then:

```59:71:src/detectors/entropy_cpu.cpp
EntropyDetector::EntropyFeatures EntropyDetector::calculateFeatures(const WindowStats& window) {
    EntropyFeatures features;
    if (window.total_packets == 0) return features;
    
    features.src_ip_entropy      = calculateEntropy(window.src_ip_counts,      window.total_packets);
    features.dst_ip_entropy      = calculateEntropy(window.dst_ip_counts,      window.total_packets);
    features.src_port_entropy    = calculateEntropy(window.src_port_counts,    window.total_packets);
    features.dst_port_entropy    = calculateEntropy(window.dst_port_counts,    window.total_packets);
    features.packet_size_entropy = calculateEntropy(window.packet_size_counts, window.total_packets);
    features.protocol_entropy    = calculateEntropy(window.protocol_counts,    window.total_packets);
    return features;
}
```

### 3.3 Entropy anomaly score

The entropy features are converted to a normalized **anomaly score** in \([0,1]\):

```81:97:src/detectors/entropy_cpu.cpp
double EntropyDetector::getAnomalyScore(const EntropyFeatures& features) {
    double avg_entropy = (features.src_ip_entropy + features.dst_ip_entropy + 
                         features.src_port_entropy + features.dst_port_entropy +
                         features.packet_size_entropy + features.protocol_entropy) / 6.0;
    
    // Assume max entropy ~ 15 bits
    double normalized = std::min(1.0, avg_entropy / 15.0);
    
    // Lower entropy = higher anomaly score
    return 1.0 - normalized;
}
```

So:

- **High entropy** (diverse traffic) ⇒ `normalized` close to 1 ⇒ anomaly score near 0 ⇒ benign.
- **Low entropy** (concentrated traffic) ⇒ `normalized` small ⇒ anomaly score near 1 ⇒ suspicious.

`DecisionEngine` uses this score with a configurable `entropy_threshold` (see `config/detection_config.json`).

---

## 4. CUSUM detector: theory and implementation

### 4.1 CUSUM concept

**CUSUM (Cumulative Sum) change detection** is a classical method for detecting shifts in the mean of a sequence:

- Maintain cumulative sums of deviations from a baseline mean.
- Subtract a **drift** parameter to avoid false positives on noise.
- Compare the running statistic to a **threshold**; if it exceeds the threshold, a change/anomaly is declared.

Mathematically, for samples \( x_t \) and baseline mean \( \mu \), a simplified one‑sided CUSUM is:

\[
S_t = \max(0, S_{t-1} + (x_t - \mu - k))
\]

where \( k \) is the drift; an alarm is triggered if \( S_t > h \) for some threshold \( h \).

### 4.2 CUSUM in this codebase

`CUSUMDetector` implements a **two‑sided** CUSUM on the **average entropy per window**:

```9:21:src/detectors/cusum_detector.h
class CUSUMDetector {
public:
    CUSUMDetector(double threshold = 5.0, double drift = 0.5, size_t window_size = 100);
    ...
    void update(double entropy_value);
    bool isAnomaly() const { return anomaly_detected_; }
    double getStatistic() const { return s_positive_ + s_negative_; }
private:
    double threshold_;
    double drift_;
    size_t window_size_;    // number of recent entropy samples for baseline
    double s_positive_;
    double s_negative_;
    double baseline_mean_;
    bool   anomaly_detected_;
    std::deque<double> recent_values_;
    void updateBaseline(double value);
    void updateCUSUM(double value);
};
```

Baseline maintenance uses a sliding window of the last `window_size_` entropy values:

```13:27:src/detectors/cusum_detector.cpp
void CUSUMDetector::updateBaseline(double value) {
    recent_values_.push_back(value);
    if (recent_values_.size() > window_size_) {
        recent_values_.pop_front();
    }
    if (!recent_values_.empty()) {
        double sum = std::accumulate(recent_values_.begin(), recent_values_.end(), 0.0);
        baseline_mean_ = sum / recent_values_.size();
    } else {
        baseline_mean_ = value;
    }
}
```

CUSUM updates and anomaly decision:

```29:41:src/detectors/cusum_detector.cpp
void CUSUMDetector::updateCUSUM(double value) {
    double deviation = value - baseline_mean_;
    
    // Positive CUSUM (increases)
    s_positive_ = std::max(0.0, s_positive_ + deviation - drift_);
    
    // Negative CUSUM (decreases)
    s_negative_ = std::max(0.0, s_negative_ - deviation - drift_);
    
    double total_stat = s_positive_ + s_negative_;
    anomaly_detected_ = (total_stat > threshold_);
}
```

The unified `update` method performs both steps per sample:

```44:47:src/detectors/cusum_detector.cpp
void CUSUMDetector::update(double entropy_value) {
    updateBaseline(entropy_value);
    updateCUSUM(entropy_value);
}
```

### 4.3 What CUSUM is tracking here

`DecisionEngine::update` feeds CUSUM with the **average of the six entropy values** for each window:

```41:50:src/detectors/decision_engine.cpp
void DecisionEngine::update(const EntropyDetector::EntropyFeatures& entropy_features,
                           double ml_probability,
                           const EntropyDetector::EntropyFeatures& current_features) {
    (void)ml_probability;
    (void)current_features;
    double avg_entropy = (entropy_features.src_ip_entropy + entropy_features.dst_ip_entropy +
                         entropy_features.src_port_entropy + entropy_features.dst_port_entropy +
                         entropy_features.packet_size_entropy + entropy_features.protocol_entropy) / 6.0;
    cusum_detector_.update(avg_entropy);
    ...
}
```

Thus:

- **Input sequence** to CUSUM = average entropy per window.
- **Baseline** = running mean of recent entropies (last `window_size_` windows).
- **Drift** = `drift_` (default 0.5), prevents small fluctuations from accumulating.
- **Threshold** = `cusum_threshold` from `config/detection_config.json` (e.g., 2.5), controls sensitivity.

`CUSUMDetector::isAnomaly()` contributes a boolean `cusum_alert` in `DecisionEngine`, and `getStatistic()` is normalized against `cusum_threshold_` for scoring.

---

## 5. ML‑based detection: training and inference

### 5.1 Attack vs benign labels (training)

Training logic in `src/ml/train_ml.py` converts dataset labels into **binary labels**:

```48:57:src/ml/train_ml.py
        if label_col:
            // Map labels: BENIGN -> 0, attack types -> 1
            df['label'] = (df[label_col] != 'BENIGN').astype(int)
            ...
        else:
            // If no Label column, infer from filename
            attack_keywords = ['ddos', 'drdos', 'syn', 'udp', 'ldap', ...]
            filename_lower = csv_file.name.lower()
            is_attack = any(keyword in filename_lower for keyword in attack_keywords)
            df['label'] = 1 if is_attack else 0
```

So the ML model is trained on:

- **0 = Benign** (explicit BENIGN label or non‑attack filename).
- **1 = Attack** (any non‑BENIGN label or filenames indicating DDoS/DRDoS etc.).

All traditional multiclass attack types are collapsed into a single **attack** class for binary classification.

### 5.2 ML feature set

At runtime, features are built from `WindowStats` and entropy values by `FeatureBuilder`:

```68:107:src/ml/feature_builder.cpp
bool FeatureBuilder::buildFeatures(const WindowStats& window,
                                  const std::vector<double>& gpu_entropy_results,
                                  std::vector<double>& feature_vector) {
    // Require 6 entropy values per window
    if (gpu_entropy_results.size() < 6) return false;
    feature_vector.clear();
    feature_vector.reserve(24);
    
    // 1. total_packets
    feature_vector.push_back(static_cast<double>(window.total_packets));
    // 2. total_bytes
    feature_vector.push_back(static_cast<double>(window.total_bytes));
    // 3. flow_duration_ms
    uint64_t window_duration_us = window.window_end_us - window.window_start_us;
    double flow_duration_ms = static_cast<double>(window_duration_us) / 1000.0;
    feature_vector.push_back(flow_duration_ms);
    // 4. flow_bytes_per_sec
    ...
    // 5. flow_packets_per_sec
    ...
    // 6. avg_packet_size
    ...
    // 7–8. packet_size_mean and std
    ...
    // 9–11. TCP SYN/FIN/RST counts
    feature_vector.push_back(static_cast<double>(window.syn_packets));
    feature_vector.push_back(static_cast<double>(window.fin_packets));
    feature_vector.push_back(static_cast<double>(window.rst_packets));
    // 12–16. Entropy features from GPU: src_ip, dst_ip, src_port, dst_port, protocol
    feature_vector.push_back(gpu_entropy_results[0]);
    feature_vector.push_back(gpu_entropy_results[1]);
    feature_vector.push_back(gpu_entropy_results[2]);
    feature_vector.push_back(gpu_entropy_results[3]);
    feature_vector.push_back(gpu_entropy_results[5]);
    // 17–20. Unique src/dst IPs and ports
    ...
    // 21–22. Top‑10 source/dest IP fractions
    ...
    // 23. Packet size coefficient of variation
    ...
    // 24. flow_count
    feature_vector.push_back(static_cast<double>(window.flow_count));
}
```

There is also a `buildFeaturesFromCPU` variant that uses `EntropyDetector::EntropyFeatures` instead of GPU entropy results; it constructs the same 24‑dimensional vector for CPU‑only operation.

This feature set mixes:

- **Volume and rate** features (total packets/bytes, bytes/sec, packets/sec).
- **Size statistics** (mean, std, coefficient of variation).
- **TCP behavior** (SYN/FIN/RST counts).
- **Diversity** (entropy features, unique IPs/ports, top‑N concentration).
- **Flow richness** (`flow_count`).

These are designed to distinguish benign vs DDoS‑like windows.

### 5.3 GPU entropy computation feeding ML

When GPU is available, entropy is computed by `GPUDetector` using OpenCL:

```105:125:src/opencl/gpu_detector.cpp
bool GPUDetector::processBatch(const std::vector<WindowStats>& windows,
                               std::vector<double>& entropy_results) {
    ...
    const size_t num_windows = windows.size();
    const size_t num_bins = 256;
    const size_t num_features = 6;  // src_ip, dst_ip, src_port, dst_port, packet_size, protocol
    ...
    cl_kernel kernel = opencl_host_.getKernel("compute_multi_entropy");
    ...
    size_t global_work_size = num_windows * num_features;
    opencl_host_.executeKernelPrepared("compute_multi_entropy", global_work_size, 0);
    ...
    // entropy_out has num_windows * num_features entropies
    for (size_t i = 0; i < num_windows; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            entropy_results.push_back(static_cast<double>(entropy_out[i * num_features + j]));
        }
    }
    return true;
}
```

These GPU‑computed entropies are both:

- Used directly in ML features (`FeatureBuilder::buildFeatures`).
- Translated into `EntropyDetector::EntropyFeatures` for entropy‑based and CUSUM/PCA detectors.

### 5.4 Inference: from features to attack probability

`MLInferenceEngine` abstracts loading scikit‑learn models via `joblib` and calling `predict_proba`:

```11:25:src/ml/inference_engine.h
class MLInferenceEngine {
public:
    ...
    bool loadModel(const std::string& model_path,
                   const std::string& preprocessor_path = "",
                   const std::vector<std::string>& module_imports = {});
    
    // Predict single sample (returns probability of attack class)
    double predict(const std::vector<double>& features);
    
    // Predict batch of samples (probability of attack class)
    std::vector<double> predictBatch(const std::vector<std::vector<double>>& features_batch);
};
```

The batch method extracts the **attack class probability (class index 1)**:

```369:376:src/ml/inference_engine.cpp
// result is a NumPy array of shape [batch_size, 2]
if (PyArray_Check(result)) {
    PyArrayObject* result_array = (PyArrayObject*)result;
    if (PyArray_NDIM(result_array) == 2 && PyArray_DIM(result_array, 1) >= 2) {
        double* data = (double*)PyArray_DATA(result_array);
        for (size_t i = 0; i < batch_size; ++i) {
            results.push_back(data[i * 2 + 1]);  // Probability of attack class
        }
    }
}
```

In the runtime batch processing in `main.cpp`, we see:

```251:270:src/main.cpp
if (ml_available && ml_engine && !ml_feature_vectors.empty()) {
    ml_probabilities = ml_engine->predictBatch(ml_feature_vectors);
} else {
    ml_probabilities.assign(window_batch.size(), 0.5);
}
...
EntropyDetector::EntropyFeatures features;
features.src_ip_entropy      = entropy_results[base_idx + 0];
features.dst_ip_entropy      = entropy_results[base_idx + 1];
features.src_port_entropy    = entropy_results[base_idx + 2];
features.dst_port_entropy    = entropy_results[base_idx + 3];
features.packet_size_entropy = entropy_results[base_idx + 4];
features.protocol_entropy    = entropy_results[base_idx + 5];
double ml_prob = (ml_idx < ml_probabilities.size()) ? ml_probabilities[ml_idx++] : 0.5;
DetectionResult result = decision_engine.detect(features, ml_prob);
```

So **per window**, the ML detector outputs `ml_probability` ∈ \([0,1]\) representing “probability of ATTACK”, which is then combined with entropy/CUSUM/PCA in the ensemble.

---

## 6. Ensemble decision engine: ATTACK vs benign

### 6.1 DetectionResult and thresholds

`DecisionEngine` defines the per‑window detection output:

```11:23:src/detectors/decision_engine.h
struct DetectionResult {
    bool   is_attack;
    double combined_score;
    double entropy_score;
    double ml_score;
    double cusum_score;
    double pca_score;
    std::string detector_triggered;  // Which detector(s) triggered
    
    DetectionResult()
        : is_attack(false), combined_score(0.0),
          entropy_score(0.0), ml_score(0.0),
          cusum_score(0.0), pca_score(0.0) {}
};
```

It is configured with four thresholds and a mode flag:

```27:31:src/detectors/decision_engine.h
DecisionEngine(double entropy_threshold = 0.5,
               double ml_threshold = 0.5,
               double cusum_threshold = 5.0,
               double pca_threshold = 0.1,
               bool use_weighted = false);
```

The actual values are loaded from `config/detection_config.json`:

```138:152:src/main.cpp
DetectionConfig detection_cfg;
if (!loadDetectionConfig(detection_config_path, detection_cfg)) { ... }
DecisionEngine decision_engine(detection_cfg.entropy_threshold,
                               detection_cfg.ml_threshold,
                               detection_cfg.cusum_threshold,
                               detection_cfg.pca_threshold,
                               detection_cfg.use_weighted);
decision_engine.setWeights(detection_cfg.w_entropy,
                           detection_cfg.w_ml,
                           detection_cfg.w_cusum,
                           detection_cfg.w_pca);
```

Example config:

```1:12:config/detection_config.json
{
  "entropy_threshold": 0.55,
  "ml_threshold": 0.45,
  "cusum_threshold": 2.5,
  "pca_threshold": 1.25,
  "use_weighted": true,
  "weights": { "entropy": 0.35, "ml": 0.45, "cusum": 0.1, "pca": 0.1 }
}
```

### 6.2 Ensemble (OR‑logic) mode

In **ensemble mode** (`use_weighted_ == false`), the decision engine triggers ATTACK if **any** detector crosses its threshold:

```65:83:src/detectors/decision_engine.cpp
DetectionResult DecisionEngine::detectEnsemble(const EntropyDetector::EntropyFeatures& entropy_features,
                                               double ml_probability) {
    DetectionResult result;
    // 1) Compute scores
    result.entropy_score = entropy_detector_.getAnomalyScore(entropy_features);
    result.ml_score      = ml_probability;
    result.cusum_score   = cusum_detector_.getStatistic() / cusum_threshold_;
    result.pca_score     = pca_detector_.getReconstructionError(entropy_features) / pca_threshold_;
    
    // 2) Individual alerts
    bool entropy_alert = result.entropy_score > entropy_threshold_;
    bool ml_alert      = result.ml_score      > ml_threshold_;
    bool cusum_alert   = cusum_detector_.isAnomaly();
    bool pca_alert     = pca_detector_.detectAnomaly(entropy_features);
    
    // 3) ATTACK if ANY triggers
    result.is_attack = entropy_alert || ml_alert || cusum_alert || pca_alert;
    ...
}
```

`detector_triggered` records which detectors fired, for interpretability (e.g., `"Entropy+ML"` or `"CUSUM"`).

### 6.3 Weighted mode

In **weighted mode** (`use_weighted_ == true`), a single **combined score** in \([0,1]\) is computed from normalized detector scores and compared to an internal threshold (0.5):

```105:122:src/detectors/decision_engine.cpp
DetectionResult DecisionEngine::detectWeighted(const EntropyDetector::EntropyFeatures& entropy_features,
                                               double ml_probability) {
    DetectionResult result;
    result.entropy_score = entropy_detector_.getAnomalyScore(entropy_features);
    result.ml_score      = ml_probability;
    result.cusum_score   = std::min(1.0, cusum_detector_.getStatistic() / cusum_threshold_);
    result.pca_score     = std::min(1.0, pca_detector_.getReconstructionError(entropy_features) / pca_threshold_);
    
    // Weighted sum of normalized scores
    result.combined_score = w_entropy_ * result.entropy_score +
                           w_ml_      * result.ml_score +
                           w_cusum_   * result.cusum_score +
                           w_pca_     * result.pca_score;
    
    // ATTACK if combined_score > 0.5
    result.is_attack = result.combined_score > 0.5;
    ...
}
```

The weights come from the config file and are normalized in `setWeights`. In this mode:

- **All detectors contribute continuously** to the risk score.
- High ML probability can be offset or reinforced by entropy/CUSUM/PCA depending on weights.

### 6.4 Runtime ATTACK vs benign behavior

`main.cpp` processes windows in batches and invokes the decision engine for each:

```257:272:src/main.cpp
DetectionResult result = decision_engine.detect(features, ml_prob);
uint64_t window_index = (window_count - window_batch.size() + i);
logResult(result, ml_prob, window_batch[i], window_index);
```

Inside `logResult`, only windows with `result.is_attack == true` are treated as ATTACK:

```191:199:src/main.cpp
auto logResult = [&](const DetectionResult& result, double ml_prob, const WindowStats& stats, uint64_t index) {
    if (!result.is_attack) {
        return;
    }
    std::cout << "ATTACK DETECTED in window " << index
              << " - Triggers: " << result.detector_triggered << std::endl;
    uint64_t timestamp_ms    = stats.window_end_us / 1000ULL;
    uint64_t window_start_ms = stats.window_start_us / 1000ULL;
    uint32_t top_src_ip      = getTopSrcIp(stats);
    logger.logAlert(timestamp_ms, window_start_ms, index, top_src_ip,
                    result.entropy_score, ml_prob,
                    result.cusum_score, result.pca_score,
                    result.combined_score, result.detector_triggered,
                    selected_model_name);
    ...
};
```

- Windows **without** `is_attack` set simply produce no alert and are implicitly **benign**.
- When an ATTACK window is detected, the top source IP is added to the blackhole list for mitigation.

---

## 7. Runtime pipeline: from packets to alerts and blocking

### 7.1 Main processing loop

The heart of the system is the loop in `main.cpp`:

```341:367:src/main.cpp
while (pcap_reader.readNextPacket(packet)) {
    if (!packet.is_valid) continue;
    
    // Drop if source or destination IP is blackholed
    if (pcap_filter.shouldDrop(packet.src_ip, packet.dst_ip)) {
        dropped_packets_total++;
        ...
        continue; // Skip this packet; don't add to window
    }
    
    packet_count++;
    
    // Add to window
    window_manager.addPacket(packet);
    
    // Close window if it exceeds window_size_us_ (calls callback)
    window_manager.checkWindow(packet.timestamp_us);
    ...
}
window_manager.closeWindow();  // Force‑close final window
if (!window_batch.empty()) {
    processCurrentBatch();
}
```

The window callback pushes closed `WindowStats` into `window_batch`; when `window_batch.size() >= batch_size`, `processCurrentBatch()` is invoked to run detection on that batch.

### 7.2 Alert to RTBH feedback loop

When a window is classified as ATTACK, the top source IP is used for blackholing:

```200:221:src/main.cpp
uint32_t top_src_ip = getTopSrcIp(stats);
...
if (top_src_ip != 0) {
    ...
    if (rtbh_controller.addIP(top_src_ip)) {
        uint64_t impacted_packets = stats.total_packets;
        uint64_t dropped_packets  = 0;
        logger.logBlocking(timestamp_ms, top_src_ip, impacted_packets, dropped_packets);
        std::cout << "  -> IP " << ip_str << " added to blackhole list (will drop future packets)" << std::endl;
    }
}
```

The RTBH controller and packet filter then ensure that **future packets** from that IP are dropped before they even participate in windowing:

```11:21:src/blocking/pcap_filter.cpp
bool PcapFilter::shouldDrop(uint32_t src_ip, uint32_t dst_ip) {
    packets_processed_++;
    if (rtbh_controller_ &&
        (rtbh_controller_->isBlackholed(src_ip) || rtbh_controller_->isBlackholed(dst_ip))) {
        packets_dropped_++;
        return true;
    }
    return false;
}
```

`RTBHController` maintains the set of blackholed IPs and persists them to `blackhole.json`:

```30:37:src/blocking/rtbh_controller.cpp
bool RTBHController::addIP(uint32_t ip) {
    std::lock_guard<std::mutex> lock(mutex_);
    blackhole_list_.insert(ip);
    return true;
}
...
bool RTBHController::isBlackholed(uint32_t ip) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blackhole_list_.find(ip) != blackhole_list_.end();
}
```

This implements a simple **detect → log → block** loop per attacking IP.

---

## 8. Configuration, thresholds, and tuning

Detection sensitivity and behavior are primarily controlled by:

- `config/detection_config.json` (detector thresholds, weights, ensemble vs weighted).
- Command‑line flags: `--window`, `--batch`, `--ml-model`, `--use-gpu` / `--use-cpu`, `--detection-config`, `--model-manifest`.

### 8.1 Detection thresholds and mode

Key fields in `detection_config.json`:

- **`entropy_threshold`**: cutoff on `entropy_score` (0–1). Higher values mean the entropy must be **very low** (stronger evidence of attack) before triggering.
- **`ml_threshold`**: cutoff on ML attack probability (0–1). Often near 0.5 by default; lowering it increases recall but also false positives.
- **`cusum_threshold`**: threshold on the CUSUM statistic; lower values make it more sensitive to changes in average entropy.
- **`pca_threshold`**: threshold on PCA reconstruction error; lower values flag more deviations from “normal” entropy patterns.
- **`use_weighted`**: `false` for OR‑ensemble, `true` for weighted combination.
- **`weights.entropy`, `weights.ml`, `weights.cusum`, `weights.pca`**: contributions in weighted mode (normalized internally).

Practical guidance:

- To **reduce false positives**, consider:
  - Increasing `ml_threshold` and `entropy_threshold`.
  - Increasing `cusum_threshold` to ignore small drifts.
  - Reducing `w_entropy`/`w_cusum` and emphasizing ML in weighted mode.
- To **increase sensitivity** (catch more attacks), consider:
  - Lowering `ml_threshold` slightly (e.g., 0.45 → 0.35).
  - Lowering `entropy_threshold` so moderately low entropy already counts as suspicious.
  - Lowering `cusum_threshold` or decreasing CUSUM `drift_` for faster change detection.

Always validate tuning decisions using:

- Offline replays (`scripts/run_experiment.sh`, `scripts/evaluate_detection.py`).
- Training/reporting artifacts (`reports/training_metrics.json`, `logs/alerts.csv`).

### 8.2 Window and batch sizes

- **`--window`**: size of each detection window in seconds (default 1).
  - Larger windows: more packets per decision, smoother statistics but slower reaction.
  - Smaller windows: faster reaction but noisier statistics; may require threshold adjustments.
- **`--batch`**: number of windows processed together by GPU and ML.
  - Affects performance but not the semantics of `is_attack`; large batches are more efficient on GPU.

---

## 9. GPU vs CPU paths and performance

The detection logic itself is the same for GPU and CPU paths; they differ only in **where entropy is computed and where ML features are derived from**:

- **GPU path**:
  - `GPUDetector::processBatch` builds histograms on the host, sends them to OpenCL, runs `compute_multi_entropy` to compute entropies for all windows and features in parallel.
  - Entropies are fed to both `FeatureBuilder::buildFeatures` (for ML) and to `DecisionEngine` (via `EntropyDetector::EntropyFeatures`).
- **CPU path**:
  - For each `WindowStats`, `EntropyDetector::calculateFeatures` computes entropy on CPU.
  - `FeatureBuilder::buildFeaturesFromCPU` uses CPU entropies to build the same 24‑feature vector; the rest of the pipeline is identical.

Correctness of GPU vs CPU entropy is validated by helper tools in `tools/`:

- `tools/test_gpu_entropy.cpp`, `tools/validate_entropy.py`, `tools/validate_gpu_correctness.py`.

If GPU initialization fails, the system falls back automatically to the CPU path with a warning, without changing detection semantics.

---

## 10. Examples: benign vs attack windows

### 10.1 Benign window (high entropy, low volume)

Conceptual characteristics:

- `total_packets` modest, low packet rate.
- Many distinct source and destination IPs, larger port diversity.
- Entropy features relatively **high** (close to their maximum) for src/dst IP and ports.
- CUSUM baseline is stable; no recent significant shift.
- ML features resemble training examples labeled BENIGN.

Detector behavior:

- `entropy_score` (1 − normalized entropy) remains low ⇒ below `entropy_threshold`.
- `ml_probability` is low (e.g., < 0.2) ⇒ below `ml_threshold`.
- CUSUM statistic stays below `cusum_threshold` ⇒ `cusum_alert == false`.
- PCA reconstruction error is modest ⇒ below `pca_threshold`.
- **Result**: `DetectionResult.is_attack == false` ⇒ window treated as **benign**, no alert, no blocking.

### 10.2 DDoS‑like window (low entropy, high volume)

Conceptual characteristics:

- `total_packets` and `flow_packets_per_sec` high.
- Most traffic from a small set of sources to a single or few victim IPs.
- Repeated ports/protocols; high SYN or UDP volume.
- Entropy features for src/dst IP and port distributions are **low**.
- CUSUM sees a sudden drop in entropy relative to its baseline.
- ML features match patterns labeled ATTACK in training data.

Detector behavior:

- `entropy_score` near 1.0 ⇒ exceeds `entropy_threshold`.
- `ml_probability` high (e.g., > 0.8) ⇒ exceeds `ml_threshold`.
- CUSUM statistic exceeds `cusum_threshold` after several consecutive low‑entropy windows ⇒ `cusum_alert == true`.
- PCA reconstruction error spikes ⇒ possibly `pca_alert == true`.
- **Ensemble mode**: any of the alerts suffices to set `is_attack = true`.
- **Weighted mode**: combined score \( w_{\text{entropy}} s_{\text{entropy}} + w_{\text{ml}} s_{\text{ml}} + ... \) exceeds 0.5.
- **Result**: ATTACK alert logged; top source IP added to blackhole list; subsequent packets from that IP are dropped by `PcapFilter`.

---

## 11. File map and cross‑references

For convenience, here is a mapping from detection concepts to key implementation files:

- **Packet ingestion & windows**
  - `src/ingest/pcap_reader.h`, `src/ingest/pcap_reader.cpp`
  - `src/ingest/window_manager.h`, `src/ingest/window_manager.cpp`
- **Entropy & statistical detectors**
  - `src/detectors/entropy_cpu.h`, `src/detectors/entropy_cpu.cpp`
  - `src/detectors/cusum_detector.h`, `src/detectors/cusum_detector.cpp`
  - `src/detectors/pca_detector.h`, `src/detectors/pca_detector.cpp`
- **ML features & models**
  - `src/ml/feature_builder.cpp`, `src/ml/feature_builder.h`
  - `src/ml/feature_extractor.py`
  - `src/ml/train_ml.py`, `src/ml/train_models.py`
  - `src/ml/inference_engine.h`, `src/ml/inference_engine.cpp`
  - Model artifacts in `models/` and manifest `models/model_manifest.json`
- **Ensemble decision & runtime glue**
  - `src/detectors/decision_engine.h`, `src/detectors/decision_engine.cpp`
  - `src/main.cpp`
- **GPU acceleration**
  - `src/opencl/gpu_detector.h`, `src/opencl/gpu_detector.cpp`
  - Kernels in `src/opencl/kernels/*.cl`
- **Blocking & logging**
  - `src/blocking/rtbh_controller.h`, `src/blocking/rtbh_controller.cpp`
  - `src/blocking/pcap_filter.h`, `src/blocking/pcap_filter.cpp`
  - `src/utils/logger.h`, `src/utils/logger.cpp`
- **Configuration & scripts**
  - `config/detection_config.json`
  - `scripts/run_experiment.sh`, `scripts/evaluate_detection.py`
  - Validation and benchmarking tools under `tools/`

Together, these components implement a **window‑based DDoS detection pipeline** that combines statistical anomaly detectors (entropy, CUSUM, PCA) with supervised ML classification to decide whether each time window of traffic is ATTACK or benign, and optionally mitigates attacks by blackholing top talker IPs.


