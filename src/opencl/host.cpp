#include "host.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>

OpenCLHost::OpenCLHost()
    : platform_(nullptr), device_(nullptr), context_(nullptr), queue_(nullptr),
      max_work_group_size_(0), initialized_(false) {
}

OpenCLHost::~OpenCLHost() {
    cleanup();
}

bool OpenCLHost::initialize(const std::string& device_name) {
    if (initialized_) {
        std::cerr << "OpenCL already initialized" << std::endl;
        return false;
    }
    
    // Find platform
    if (!findPlatform()) {
        std::cerr << "Failed to find OpenCL platform" << std::endl;
        return false;
    }
    
    // Find device
    if (!findDevice(device_name)) {
        std::cerr << "Failed to find device: " << device_name << std::endl;
        return false;
    }
    
    // Create context
    if (!createContext()) {
        std::cerr << "Failed to create context" << std::endl;
        return false;
    }
    
    // Create command queue with profiling
    if (!createCommandQueue()) {
        std::cerr << "Failed to create command queue" << std::endl;
        return false;
    }
    
    // Get device info
    cl_ulong max_work_group_size = 0;
    clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, nullptr);
    max_work_group_size_ = static_cast<size_t>(max_work_group_size);
    
    char name[256];
    size_t name_len;
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, &name_len);
    device_name_ = std::string(name, name_len);
    
    std::cout << "OpenCL initialized successfully" << std::endl;
    std::cout << "  Device: " << device_name_ << std::endl;
    std::cout << "  Max work group size: " << max_work_group_size_ << std::endl;
    
    initialized_ = true;
    return true;
}

bool OpenCLHost::findPlatform() {
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: clGetPlatformIDs failed with error code: " << err << std::endl;
        return false;
    }
    if (num_platforms == 0) {
        std::cerr << "ERROR: No OpenCL platforms found. Install OpenCL drivers (e.g., nvidia-opencl-dev, ocl-icd-opencl-dev)" << std::endl;
        std::cerr << "  Run 'clinfo' to list available platforms/devices" << std::endl;
        return false;
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: Failed to get platform IDs: " << err << std::endl;
        return false;
    }
    
    std::cerr << "Found " << num_platforms << " OpenCL platform(s):" << std::endl;
    // Try to find preferred platform (NVIDIA, Intel, AMD, etc.)
    cl_platform_id preferred_platform = nullptr;
    int preferred_priority = -1;
    
    for (size_t i = 0; i < platforms.size(); ++i) {
        cl_platform_id p = platforms[i];
        char vendor[256] = {0};
        char name[256] = {0};
        clGetPlatformInfo(p, CL_PLATFORM_VENDOR, sizeof(vendor), vendor, nullptr);
        clGetPlatformInfo(p, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        std::cerr << "  Platform " << i << ": " << name << " (vendor: " << vendor << ")" << std::endl;
        
        // Priority: NVIDIA > Intel > AMD > Others
        int priority = 0;
        if (strstr(vendor, "NVIDIA") != nullptr) {
            priority = 3;
        } else if (strstr(vendor, "Intel") != nullptr) {
            priority = 2;
        } else if (strstr(vendor, "AMD") != nullptr || strstr(vendor, "Advanced Micro Devices") != nullptr) {
            priority = 1;
        }
        
        if (priority > preferred_priority) {
            preferred_platform = p;
            preferred_priority = priority;
        }
    }
    
    // Use preferred platform or fallback to first
    if (preferred_platform != nullptr) {
        platform_ = preferred_platform;
        char name[256] = {0};
        clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        std::cerr << "  -> Selected platform: " << name << std::endl;
    } else {
        platform_ = platforms[0];
        char name[256] = {0};
        clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(name), name, nullptr);
        std::cerr << "  -> Fallback to first platform: " << name << std::endl;
    }
    return true;
}

bool OpenCLHost::findDevice(const std::string& device_name) {
    cl_uint num_devices = 0;
    cl_int err = CL_SUCCESS;
    
    // Try GPU first
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    cl_device_type device_type = CL_DEVICE_TYPE_GPU;
    
    // If no GPU found, try CPU as fallback
    if (err != CL_SUCCESS || num_devices == 0) {
        std::cerr << "  No GPU devices found, trying CPU OpenCL..." << std::endl;
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
        device_type = CL_DEVICE_TYPE_CPU;
    }
    
    // If still no devices, try ALL device types
    if (err != CL_SUCCESS || num_devices == 0) {
        std::cerr << "  No CPU devices found, trying ALL device types..." << std::endl;
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        device_type = CL_DEVICE_TYPE_ALL;
    }
    
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: clGetDeviceIDs failed: " << err << std::endl;
        std::cerr << "  Check OpenCL drivers are installed (e.g., nvidia-opencl-dev, ocl-icd-opencl-dev)" << std::endl;
        return false;
    }
    if (num_devices == 0) {
        std::cerr << "ERROR: No OpenCL devices found on platform" << std::endl;
        std::cerr << "  Run 'clinfo -l' to list available devices" << std::endl;
        return false;
    }
    
    std::vector<cl_device_id> devices(num_devices);
    err = clGetDeviceIDs(platform_, device_type, num_devices, devices.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: Failed to get device IDs: " << err << std::endl;
        return false;
    }
    
    std::string device_type_str = (device_type == CL_DEVICE_TYPE_GPU) ? "GPU" : 
                                   (device_type == CL_DEVICE_TYPE_CPU) ? "CPU" : "ALL";
    std::cerr << "Found " << num_devices << " " << device_type_str << " device(s):" << std::endl;
    // Find device by name
    for (size_t i = 0; i < devices.size(); ++i) {
        cl_device_id d = devices[i];
        char name[256] = {0};
        size_t name_len;
        clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, &name_len);
        std::string dev_name(name, name_len);
        std::cerr << "  Device " << i << ": " << dev_name << std::endl;
        
        if (dev_name.find(device_name) != std::string::npos) {
            device_ = d;
            std::cerr << "  -> Selected device matching '" << device_name << "'" << std::endl;
            return true;
        }
    }
    
    // Fallback to first device
    device_ = devices[0];
    char name[256] = {0};
    size_t name_len;
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(name), name, &name_len);
    std::cerr << "  -> Fallback to first device: " << std::string(name, name_len) << std::endl;
    return true;
}

bool OpenCLHost::createContext() {
    cl_int err;
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "ERROR: Failed to create OpenCL context: " << err << std::endl;
        std::cerr << "  Common causes:" << std::endl;
        std::cerr << "    - GPU drivers not installed or outdated" << std::endl;
        std::cerr << "    - OpenCL runtime not available (install ocl-icd-opencl-dev)" << std::endl;
        std::cerr << "    - Insufficient permissions (try running with sudo or check udev rules)" << std::endl;
        std::cerr << "  Run 'clinfo' to verify OpenCL setup" << std::endl;
    }
    return err == CL_SUCCESS;
}

bool OpenCLHost::createCommandQueue() {
    cl_int err;
    // Use deprecated clCreateCommandQueue for compatibility (OpenCL 1.x)
    // TODO: Update to clCreateCommandQueueWithProperties for OpenCL 2.0+
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    queue_ = clCreateCommandQueue(context_, device_, CL_QUEUE_PROFILING_ENABLE, &err);
    #pragma GCC diagnostic pop
    return err == CL_SUCCESS;
}

std::string OpenCLHost::readKernelSource(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << std::endl;
        return "";
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool OpenCLHost::loadKernel(const std::string& kernel_file, const std::string& kernel_name) {
    if (!initialized_) {
        std::cerr << "OpenCL not initialized" << std::endl;
        return false;
    }
    
    // Read kernel source
    std::string source = readKernelSource(kernel_file);
    if (source.empty()) {
        return false;
    }
    
    const char* source_str = source.c_str();
    size_t source_len = source.length();
    
    // Create program
    cl_int err;
    cl_program program = clCreateProgramWithSource(context_, 1, &source_str, &source_len, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create program" << std::endl;
        return false;
    }
    
    // Build program
    err = clBuildProgram(program, 1, &device_, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        char build_log[4096];
        clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, sizeof(build_log), build_log, nullptr);
        std::cerr << "Build error: " << build_log << std::endl;
        clReleaseProgram(program);
        return false;
    }
    
    // Create kernel
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel: " << kernel_name << std::endl;
        clReleaseProgram(program);
        return false;
    }
    
    programs_.push_back(program);
    kernels_.push_back(kernel);
    kernel_names_.push_back(kernel_name);
    kernel_events_.push_back(nullptr);
    
    std::cout << "Loaded kernel: " << kernel_name << std::endl;
    return true;
}

cl_mem OpenCLHost::createBuffer(cl_mem_flags flags, size_t size, void* host_ptr) {
    if (!initialized_) return nullptr;
    
    cl_int err;
    cl_mem buffer = clCreateBuffer(context_, flags, size, host_ptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create buffer" << std::endl;
        return nullptr;
    }
    
    return buffer;
}

bool OpenCLHost::writeBuffer(cl_mem buffer, size_t size, void* data, bool blocking) {
    if (!initialized_ || !buffer) return false;
    
    cl_int err = clEnqueueWriteBuffer(queue_, buffer, blocking ? CL_TRUE : CL_FALSE,
                                      0, size, data, 0, nullptr, nullptr);
    return err == CL_SUCCESS;
}

bool OpenCLHost::readBuffer(cl_mem buffer, size_t size, void* data, bool blocking) {
    if (!initialized_ || !buffer) return false;
    
    cl_int err = clEnqueueReadBuffer(queue_, buffer, blocking ? CL_TRUE : CL_FALSE,
                                     0, size, data, 0, nullptr, nullptr);
    return err == CL_SUCCESS;
}

bool OpenCLHost::writeBufferAsync(cl_mem buffer, size_t size, void* data, cl_event* event) {
    if (!initialized_ || !buffer || !event) return false;
    
    cl_int err = clEnqueueWriteBuffer(queue_, buffer, CL_FALSE,
                                      0, size, data, 0, nullptr, event);
    return err == CL_SUCCESS;
}

void OpenCLHost::releaseBuffer(cl_mem buffer) {
    if (buffer) {
        clReleaseMemObject(buffer);
    }
}

cl_kernel OpenCLHost::getKernel(const std::string& kernel_name) {
    for (size_t i = 0; i < kernel_names_.size(); ++i) {
        if (kernel_names_[i] == kernel_name) {
            return kernels_[i];
        }
    }
    return nullptr;
}

bool OpenCLHost::executeKernel(const std::string& kernel_name,
                               size_t global_work_size,
                               size_t local_work_size,
                               const std::vector<cl_mem>& args) {
    if (!initialized_) return false;
    
    cl_kernel kernel = getKernel(kernel_name);
    if (!kernel) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }
    
    // Set kernel arguments
    for (size_t i = 0; i < args.size(); ++i) {
        cl_int err = clSetKernelArg(kernel, i, sizeof(cl_mem), &args[i]);
        if (err != CL_SUCCESS) {
            std::cerr << "Failed to set kernel argument " << i << std::endl;
            return false;
        }
    }
    
    // Find event index
    size_t event_idx = 0;
    for (size_t i = 0; i < kernel_names_.size(); ++i) {
        if (kernel_names_[i] == kernel_name) {
            event_idx = i;
            break;
        }
    }
    
    // Release old event if exists
    if (kernel_events_[event_idx]) {
        clReleaseEvent(kernel_events_[event_idx]);
    }
    
    // Execute kernel
    size_t local_size = local_work_size > 0 ? local_work_size : max_work_group_size_;
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr,
                                        &global_work_size, local_work_size > 0 ? &local_size : nullptr,
                                        0, nullptr, &event);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to execute kernel" << std::endl;
        return false;
    }
    
    kernel_events_[event_idx] = event;
    return true;
}

bool OpenCLHost::executeKernelPrepared(const std::string& kernel_name,
                                       size_t global_work_size,
                                       size_t local_work_size) {
    if (!initialized_) return false;
    
    cl_kernel kernel = getKernel(kernel_name);
    if (!kernel) {
        std::cerr << "Kernel not found: " << kernel_name << std::endl;
        return false;
    }
    
    // Find event index
    size_t event_idx = 0;
    bool found = false;
    for (size_t i = 0; i < kernel_names_.size(); ++i) {
        if (kernel_names_[i] == kernel_name) {
            event_idx = i;
            found = true;
            break;
        }
    }
    
    if (!found) {
        std::cerr << "Kernel not found in events list: " << kernel_name << std::endl;
        return false;
    }
    
    // Release old event if exists
    if (kernel_events_[event_idx]) {
        clReleaseEvent(kernel_events_[event_idx]);
    }
    
    // Execute kernel (arguments already set)
    size_t local_size = local_work_size > 0 ? local_work_size : max_work_group_size_;
    cl_event event;
    cl_int err = clEnqueueNDRangeKernel(queue_, kernel, 1, nullptr,
                                        &global_work_size, local_work_size > 0 ? &local_size : nullptr,
                                        0, nullptr, &event);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to execute kernel" << std::endl;
        return false;
    }
    
    kernel_events_[event_idx] = event;
    return true;
}

void OpenCLHost::finish() {
    if (initialized_ && queue_) {
        clFinish(queue_);
    }
}

double OpenCLHost::getKernelTime(const std::string& kernel_name) {
    size_t event_idx = 0;
    bool found = false;
    for (size_t i = 0; i < kernel_names_.size(); ++i) {
        if (kernel_names_[i] == kernel_name) {
            event_idx = i;
            found = true;
            break;
        }
    }
    
    if (!found || !kernel_events_[event_idx]) {
        return 0.0;
    }
    
    cl_event event = kernel_events_[event_idx];
    clWaitForEvents(1, &event);
    
    cl_ulong start = 0;
    cl_ulong end = 0;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                            sizeof(start), &start, nullptr);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                            sizeof(end), &end, nullptr);
    
    if (end <= start) {
        return 0.0;
    }
    return static_cast<double>(end - start) * 1e-6;  // Convert to milliseconds
}

void OpenCLHost::cleanup() {
    // Release events
    for (cl_event event : kernel_events_) {
        if (event) {
            clReleaseEvent(event);
        }
    }
    kernel_events_.clear();
    
    // Release kernels
    for (cl_kernel kernel : kernels_) {
        if (kernel) {
            clReleaseKernel(kernel);
        }
    }
    kernels_.clear();
    kernel_names_.clear();
    
    // Release programs
    for (cl_program program : programs_) {
        if (program) {
            clReleaseProgram(program);
        }
    }
    programs_.clear();
    
    // Release queue
    if (queue_) {
        clFinish(queue_);
        clReleaseCommandQueue(queue_);
        queue_ = nullptr;
    }
    
    // Release context
    if (context_) {
        clReleaseContext(context_);
        context_ = nullptr;
    }
    
    initialized_ = false;
}

void OpenCLHost::checkError(cl_int err, const std::string& operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error in " << operation << ": " << err << std::endl;
    }
}

