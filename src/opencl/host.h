#ifndef OPENCL_HOST_H
#define OPENCL_HOST_H

#include <CL/cl.h>
#include <string>
#include <vector>
#include <cstdint>

class OpenCLHost {
public:
    OpenCLHost();
    ~OpenCLHost();
    
    // Initialize OpenCL context and device
    bool initialize(const std::string& device_name = "NVIDIA");
    
    // Load kernel from source file
    bool loadKernel(const std::string& kernel_file, const std::string& kernel_name);
    
    // Create buffer
    cl_mem createBuffer(cl_mem_flags flags, size_t size, void* host_ptr = nullptr);
    
    // Write buffer (async)
    bool writeBuffer(cl_mem buffer, size_t size, void* data, bool blocking = false);
    
    // Read buffer (async)
    bool readBuffer(cl_mem buffer, size_t size, void* data, bool blocking = false);
    
    // Execute kernel
    bool executeKernel(const std::string& kernel_name,
                      size_t global_work_size,
                      size_t local_work_size = 0,
                      const std::vector<cl_mem>& args = {});
    
    // Wait for completion
    void finish();
    
    // Get kernel execution time (in milliseconds)
    double getKernelTime(const std::string& kernel_name);
    
    // Cleanup
    void cleanup();
    
    // Check if initialized
    bool isInitialized() const { return initialized_; }
    
    // Get device info
    std::string getDeviceName() const { return device_name_; }
    size_t getMaxWorkGroupSize() const { return max_work_group_size_; }

private:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    
    std::string device_name_;
    size_t max_work_group_size_;
    
    bool initialized_;
    
    std::vector<cl_program> programs_;
    std::vector<cl_kernel> kernels_;
    std::vector<std::string> kernel_names_;
    std::vector<cl_event> kernel_events_;
    
    // Helper functions
    bool findPlatform();
    bool findDevice(const std::string& device_name);
    bool createContext();
    bool createCommandQueue();
    cl_kernel getKernel(const std::string& kernel_name);
    std::string readKernelSource(const std::string& filename);
    void checkError(cl_int err, const std::string& operation);
};

#endif // OPENCL_HOST_H

