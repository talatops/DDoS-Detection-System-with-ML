#ifndef RESOURCE_MONITOR_H
#define RESOURCE_MONITOR_H

#include <cstdint>

struct ResourceMetrics {
    double cpu_percent;
    double gpu_percent;
    double memory_mb;
    double pps_in;
    double pps_processed;
    
    ResourceMetrics() : cpu_percent(0), gpu_percent(0), memory_mb(0),
                       pps_in(0), pps_processed(0) {}
};

class ResourceMonitor {
public:
    ResourceMonitor();
    ~ResourceMonitor();
    
    // Get current resource metrics
    ResourceMetrics getMetrics();
    
    // Update packet counters
    void updatePacketCount(uint64_t packets_in, uint64_t packets_processed);

private:
    uint64_t packets_in_total_;
    uint64_t packets_processed_total_;
    uint64_t last_update_time_;
    
    double getCPUPercent();
    double getGPUPercent();
    double getMemoryMB();
};

#endif // RESOURCE_MONITOR_H

