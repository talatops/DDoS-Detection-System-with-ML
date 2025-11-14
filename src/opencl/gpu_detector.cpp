#include "gpu_detector.h"
#include <algorithm>
#include <cmath>

GPUDetector::GPUDetector()
    : initialized_(false), batch_size_(256), last_kernel_time_ms_(0.0),
      counts_buffer_(nullptr), totals_buffer_(nullptr), entropy_buffer_(nullptr) {
}

GPUDetector::~GPUDetector() {
    cleanupBuffers();
}

bool GPUDetector::initialize() {
    if (initialized_) return true;
    
    // Initialize OpenCL
    if (!opencl_host_.initialize("NVIDIA")) {
        return false;
    }
    
    // Load entropy kernel
    if (!opencl_host_.loadKernel("src/opencl/kernels/entropy.cl", "compute_entropy")) {
        return false;
    }
    
    initialized_ = true;
    return true;
}

bool GPUDetector::prepareBatchData(const std::vector<WindowStats>& windows,
                                   std::vector<uint32_t>& counts,
                                   std::vector<uint32_t>& totals) {
    const size_t num_bins = 256;  // Histogram bins per feature
    const size_t num_features = 6;  // 6 entropy features
    
    counts.clear();
    totals.clear();
    
    for (const auto& window : windows) {
        totals.push_back(window.total_packets);
        
        // Flatten histograms into counts array
        // Format: [window0_feature0_bin0, window0_feature0_bin1, ..., window0_feature1_bin0, ...]
        
        // Feature 0: src_ip_entropy (use src_ip_counts)
        for (size_t i = 0; i < num_bins; ++i) {
            counts.push_back(0);  // Placeholder - would need to bin IP addresses
        }
        
        // Feature 1: dst_ip_entropy
        for (size_t i = 0; i < num_bins; ++i) {
            counts.push_back(0);
        }
        
        // Feature 2: src_port_entropy
        for (const auto& pair : window.src_port_counts) {
            size_t bin = pair.first % num_bins;
            if (bin < num_bins) {
                counts[counts.size() - num_bins + bin] += pair.second;
            }
        }
        // Fill remaining bins with zeros
        size_t base_idx = counts.size() - num_bins;
        for (size_t i = 0; i < num_bins; ++i) {
            if (counts.size() <= base_idx + i) {
                counts.push_back(0);
            }
        }
        
        // Feature 3: dst_port_entropy
        for (size_t i = 0; i < num_bins; ++i) {
            counts.push_back(0);
        }
        base_idx = counts.size() - num_bins;
        for (const auto& pair : window.dst_port_counts) {
            size_t bin = pair.first % num_bins;
            if (bin < num_bins && base_idx + bin < counts.size()) {
                counts[base_idx + bin] += pair.second;
            }
        }
        
        // Feature 4: packet_size_entropy
        for (size_t i = 0; i < num_bins; ++i) {
            counts.push_back(0);
        }
        base_idx = counts.size() - num_bins;
        for (const auto& pair : window.packet_size_counts) {
            size_t bin = (pair.first < num_bins) ? pair.first : (pair.first % num_bins);
            if (base_idx + bin < counts.size()) {
                counts[base_idx + bin] += pair.second;
            }
        }
        
        // Feature 5: protocol_entropy
        for (size_t i = 0; i < num_bins; ++i) {
            counts.push_back(0);
        }
        base_idx = counts.size() - num_bins;
        for (const auto& pair : window.protocol_counts) {
            size_t bin = pair.first % num_bins;
            if (base_idx + bin < counts.size()) {
                counts[base_idx + bin] += pair.second;
            }
        }
    }
    
    return true;
}

bool GPUDetector::processBatch(const std::vector<WindowStats>& windows,
                               std::vector<double>& entropy_results) {
    if (!initialized_ || windows.empty()) {
        return false;
    }
    
    // Prepare data
    std::vector<uint32_t> counts, totals;
    if (!prepareBatchData(windows, counts, totals)) {
        return false;
    }
    
    const size_t num_windows = windows.size();
    const size_t num_bins = 256;
    const size_t num_features = 6;
    
    // Create or resize buffers
    size_t counts_size = num_windows * num_features * num_bins * sizeof(uint32_t);
    size_t totals_size = num_windows * sizeof(uint32_t);
    size_t entropy_size = num_windows * num_features * sizeof(float);
    
    if (!counts_buffer_) {
        counts_buffer_ = opencl_host_.createBuffer(CL_MEM_READ_ONLY, counts_size);
        totals_buffer_ = opencl_host_.createBuffer(CL_MEM_READ_ONLY, totals_size);
        entropy_buffer_ = opencl_host_.createBuffer(CL_MEM_WRITE_ONLY, entropy_size);
    }
    
    // Write data to GPU (async)
    if (!opencl_host_.writeBuffer(counts_buffer_, counts_size, counts.data(), false)) {
        return false;
    }
    if (!opencl_host_.writeBuffer(totals_buffer_, totals_size, totals.data(), false)) {
        return false;
    }
    
    // Execute kernel
    std::vector<cl_mem> args = {counts_buffer_, totals_buffer_, entropy_buffer_};
    if (!opencl_host_.executeKernel("compute_multi_entropy",
                                    num_windows * num_features,
                                    0,  // Let OpenCL choose local size
                                    args)) {
        return false;
    }
    
    // Read results (blocking)
    std::vector<float> entropy_out(num_windows * num_features);
    if (!opencl_host_.readBuffer(entropy_buffer_, entropy_size, entropy_out.data(), true)) {
        return false;
    }
    
    // Get kernel time
    last_kernel_time_ms_ = opencl_host_.getKernelTime("compute_multi_entropy");
    
    // Process results (average entropy per window)
    entropy_results.clear();
    for (size_t i = 0; i < num_windows; ++i) {
        double avg_entropy = 0.0;
        for (size_t j = 0; j < num_features; ++j) {
            avg_entropy += entropy_out[i * num_features + j];
        }
        avg_entropy /= num_features;
        entropy_results.push_back(avg_entropy);
    }
    
    return true;
}

void GPUDetector::cleanupBuffers() {
    // Buffers will be released when OpenCL context is destroyed
    counts_buffer_ = nullptr;
    totals_buffer_ = nullptr;
    entropy_buffer_ = nullptr;
}

