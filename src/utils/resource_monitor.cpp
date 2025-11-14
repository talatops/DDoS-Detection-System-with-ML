#include "resource_monitor.h"
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/resource.h>
#include <cstring>

ResourceMonitor::ResourceMonitor()
    : packets_in_total_(0), packets_processed_total_(0), last_update_time_(0) {
}

ResourceMonitor::~ResourceMonitor() {
}

double ResourceMonitor::getCPUPercent() {
    // Simplified CPU usage calculation
    // In production, use /proc/stat or system calls
    static double last_cpu = 0.0;
    // Placeholder - would read from /proc/stat
    return last_cpu + (rand() % 10);  // Dummy value
}

double ResourceMonitor::getGPUPercent() {
    // Read from nvidia-smi or NVML
    // Simplified placeholder
    static double last_gpu = 0.0;
    return last_gpu + (rand() % 5);  // Dummy value
}

double ResourceMonitor::getMemoryMB() {
    // Read from /proc/meminfo or sysinfo
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemAvailable:") == 0) {
                std::istringstream iss(line);
                std::string label, value, unit;
                iss >> label >> value >> unit;
                return std::stod(value) / 1024.0;  // Convert KB to MB
            }
        }
    }
    return 0.0;
}

void ResourceMonitor::updatePacketCount(uint64_t packets_in, uint64_t packets_processed) {
    packets_in_total_ = packets_in;
    packets_processed_total_ = packets_processed;
}

ResourceMetrics ResourceMonitor::getMetrics() {
    ResourceMetrics metrics;
    metrics.cpu_percent = getCPUPercent();
    metrics.gpu_percent = getGPUPercent();
    metrics.memory_mb = getMemoryMB();
    metrics.pps_in = packets_in_total_;  // Would calculate rate
    metrics.pps_processed = packets_processed_total_;
    return metrics;
}

