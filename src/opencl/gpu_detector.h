#ifndef GPU_DETECTOR_H
#define GPU_DETECTOR_H

#include <vector>
#include <cstdint>
#include "opencl/host.h"
#include "ingest/window_manager.h"

class GPUDetector {
public:
    GPUDetector();
    ~GPUDetector();
    
    // Initialize GPU detector
    bool initialize();
    
    // Process batch of windows on GPU
    bool processBatch(const std::vector<WindowStats>& windows,
                     std::vector<double>& entropy_results);
    
    // Get kernel execution time
    double getKernelTime() const { return last_kernel_time_ms_; }
    
    // Get batch size
    size_t getBatchSize() const { return batch_size_; }
    void setBatchSize(size_t size) { batch_size_ = size; }

private:
    OpenCLHost opencl_host_;
    bool initialized_;
    size_t batch_size_;
    double last_kernel_time_ms_;
    
    // Buffers
    cl_mem counts_buffer_;
    cl_mem totals_buffer_;
    cl_mem entropy_buffer_;
    size_t counts_buffer_size_;
    size_t totals_buffer_size_;
    size_t entropy_buffer_size_;
    
    // Prepare data for GPU
    bool prepareBatchData(const std::vector<WindowStats>& windows,
                         std::vector<uint32_t>& counts,
                         std::vector<uint32_t>& totals);
    
    // Cleanup buffers
    void cleanupBuffers();
};

#endif // GPU_DETECTOR_H

