#ifndef METRICS_COLLECTOR_H
#define METRICS_COLLECTOR_H

#include <string>
#include <thread>
#include "utils/resource_monitor.h"
#include "utils/logger.h"

class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();
    
    // Start collecting metrics
    bool start(const std::string& output_dir = "logs");
    
    // Stop collecting
    void stop();
    
    // Update packet counters
    void updatePacketCount(uint64_t packets_in, uint64_t packets_processed);

private:
    bool running_;
    std::thread* collector_thread_;
    ResourceMonitor monitor_;
    Logger logger_;
    
    void collectLoop();
};

#endif // METRICS_COLLECTOR_H

