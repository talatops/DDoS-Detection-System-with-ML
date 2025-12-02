#include "gpu_detector.h"
#include <algorithm>
#include <cmath>
#include <CL/cl.h>
#include <iostream>

GPUDetector::GPUDetector()
    : initialized_(false), batch_size_(256), last_kernel_time_ms_(0.0),
      counts_buffer_(nullptr), totals_buffer_(nullptr), entropy_buffer_(nullptr),
      counts_buffer_size_(0), totals_buffer_size_(0), entropy_buffer_size_(0) {
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
    if (!opencl_host_.loadKernel("src/opencl/kernels/entropy.cl", "compute_multi_entropy")) {
        return false;
    }
    
    // Load feature extraction kernels (optional, for future use)
    opencl_host_.loadKernel("src/opencl/kernels/feature_extract.cl", "build_ip_histogram");
    
    initialized_ = true;
    return true;
}

bool GPUDetector::prepareBatchData(const std::vector<WindowStats>& windows,
                                   std::vector<uint32_t>& counts,
                                   std::vector<uint32_t>& totals) {
    const size_t num_bins = 256;  // Histogram bins per feature
    // 6 entropy features: src_ip, dst_ip, src_port, dst_port, packet_size, protocol
    
    counts.clear();
    totals.clear();
    
    for (const auto& window : windows) {
        totals.push_back(window.total_packets);
        
        // Flatten histograms into counts array
        // Format: [window0_feature0_bin0, window0_feature0_bin1, ..., window0_feature1_bin0, ...]
        // Each feature has num_bins bins
        
        // Feature 0: src_ip_entropy - bin IP addresses using hash
        std::vector<uint32_t> src_ip_bins(num_bins, 0);
        for (const auto& pair : window.src_ip_counts) {
            size_t bin = pair.first % num_bins;  // Hash-based binning
            src_ip_bins[bin] += pair.second;
        }
        counts.insert(counts.end(), src_ip_bins.begin(), src_ip_bins.end());
        
        // Feature 1: dst_ip_entropy
        std::vector<uint32_t> dst_ip_bins(num_bins, 0);
        for (const auto& pair : window.dst_ip_counts) {
            size_t bin = pair.first % num_bins;
            dst_ip_bins[bin] += pair.second;
        }
        counts.insert(counts.end(), dst_ip_bins.begin(), dst_ip_bins.end());
        
        // Feature 2: src_port_entropy
        std::vector<uint32_t> src_port_bins(num_bins, 0);
        for (const auto& pair : window.src_port_counts) {
            size_t bin = pair.first % num_bins;
            src_port_bins[bin] += pair.second;
        }
        counts.insert(counts.end(), src_port_bins.begin(), src_port_bins.end());
        
        // Feature 3: dst_port_entropy
        std::vector<uint32_t> dst_port_bins(num_bins, 0);
        for (const auto& pair : window.dst_port_counts) {
            size_t bin = pair.first % num_bins;
            dst_port_bins[bin] += pair.second;
        }
        counts.insert(counts.end(), dst_port_bins.begin(), dst_port_bins.end());
        
        // Feature 4: packet_size_entropy
        std::vector<uint32_t> packet_size_bins(num_bins, 0);
        for (const auto& pair : window.packet_size_counts) {
            size_t bin = (pair.first < num_bins) ? pair.first : (pair.first % num_bins);
            packet_size_bins[bin] += pair.second;
        }
        counts.insert(counts.end(), packet_size_bins.begin(), packet_size_bins.end());
        
        // Feature 5: protocol_entropy
        std::vector<uint32_t> protocol_bins(num_bins, 0);
        for (const auto& pair : window.protocol_counts) {
            size_t bin = pair.first % num_bins;
            protocol_bins[bin] += pair.second;
        }
        counts.insert(counts.end(), protocol_bins.begin(), protocol_bins.end());
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
    
    // Calculate buffer sizes
    size_t counts_size = num_windows * num_features * num_bins * sizeof(uint32_t);
    size_t totals_size = num_windows * sizeof(uint32_t);
    size_t entropy_size = num_windows * num_features * sizeof(float);
    
    // Create or resize buffers if needed
    if (!counts_buffer_ || counts_buffer_size_ < counts_size) {
        if (counts_buffer_) {
            opencl_host_.releaseBuffer(counts_buffer_);
        }
        counts_buffer_ = opencl_host_.createBuffer(CL_MEM_READ_ONLY, counts_size);
        if (!counts_buffer_) {
            std::cerr << "Failed to create counts buffer" << std::endl;
            return false;
        }
        counts_buffer_size_ = counts_size;
    }
    
    if (!totals_buffer_ || totals_buffer_size_ < totals_size) {
        if (totals_buffer_) {
            opencl_host_.releaseBuffer(totals_buffer_);
        }
        totals_buffer_ = opencl_host_.createBuffer(CL_MEM_READ_ONLY, totals_size);
        if (!totals_buffer_) {
            std::cerr << "Failed to create totals buffer" << std::endl;
            return false;
        }
        totals_buffer_size_ = totals_size;
    }
    
    if (!entropy_buffer_ || entropy_buffer_size_ < entropy_size) {
        if (entropy_buffer_) {
            opencl_host_.releaseBuffer(entropy_buffer_);
        }
        entropy_buffer_ = opencl_host_.createBuffer(CL_MEM_WRITE_ONLY, entropy_size);
        if (!entropy_buffer_) {
            std::cerr << "Failed to create entropy buffer" << std::endl;
            return false;
        }
        entropy_buffer_size_ = entropy_size;
    }
    
    // Write data to GPU (async with events)
    cl_event write_counts_event = nullptr;
    cl_event write_totals_event = nullptr;
    
    if (!opencl_host_.writeBufferAsync(counts_buffer_, counts_size, counts.data(), &write_counts_event)) {
        return false;
    }
    if (!opencl_host_.writeBufferAsync(totals_buffer_, totals_size, totals.data(), &write_totals_event)) {
        clReleaseEvent(write_counts_event);
        return false;
    }
    
    // Wait for writes to complete before launching kernel
    clWaitForEvents(1, &write_counts_event);
    clWaitForEvents(1, &write_totals_event);
    clReleaseEvent(write_counts_event);
    clReleaseEvent(write_totals_event);
    
    // Get kernel and set arguments manually (num_bins and num_features as constants)
    cl_kernel kernel = opencl_host_.getKernel("compute_multi_entropy");
    if (!kernel) {
        std::cerr << "Kernel not found: compute_multi_entropy" << std::endl;
        return false;
    }
    
    cl_int err;
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &counts_buffer_);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg 0" << std::endl;
        return false;
    }
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &totals_buffer_);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg 1" << std::endl;
        return false;
    }
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &entropy_buffer_);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg 2" << std::endl;
        return false;
    }
    uint32_t num_bins_arg = static_cast<uint32_t>(num_bins);
    err = clSetKernelArg(kernel, 3, sizeof(uint32_t), &num_bins_arg);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg 3" << std::endl;
        return false;
    }
    uint32_t num_features_arg = static_cast<uint32_t>(num_features);
    err = clSetKernelArg(kernel, 4, sizeof(uint32_t), &num_features_arg);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to set kernel arg 4" << std::endl;
        return false;
    }
    
    // Execute kernel with event profiling (arguments already set manually)
    size_t global_work_size = num_windows * num_features;
    if (!opencl_host_.executeKernelPrepared("compute_multi_entropy", global_work_size, 0)) {
        return false;
    }
    
    // Get kernel execution time from event profiling
    last_kernel_time_ms_ = opencl_host_.getKernelTime("compute_multi_entropy");
    
    // Read results (blocking - wait for kernel to complete)
    std::vector<float> entropy_out(num_windows * num_features);
    if (!opencl_host_.readBuffer(entropy_buffer_, entropy_size, entropy_out.data(), true)) {
        return false;
    }
    
    // Process results - return individual entropy values per feature per window
    entropy_results.clear();
    entropy_results.reserve(num_windows * num_features);
    for (size_t i = 0; i < num_windows; ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            entropy_results.push_back(static_cast<double>(entropy_out[i * num_features + j]));
        }
    }
    
    return true;
}

void GPUDetector::cleanupBuffers() {
    if (counts_buffer_) {
        opencl_host_.releaseBuffer(counts_buffer_);
        counts_buffer_ = nullptr;
    }
    if (totals_buffer_) {
        opencl_host_.releaseBuffer(totals_buffer_);
        totals_buffer_ = nullptr;
    }
    if (entropy_buffer_) {
        opencl_host_.releaseBuffer(entropy_buffer_);
        entropy_buffer_ = nullptr;
    }
    counts_buffer_size_ = 0;
    totals_buffer_size_ = 0;
    entropy_buffer_size_ = 0;
}

