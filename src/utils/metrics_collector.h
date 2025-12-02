#ifndef METRICS_COLLECTOR_H
#define METRICS_COLLECTOR_H

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include "utils/resource_monitor.h"
#include "utils/logger.h"

class MetricsCollector {
public:
    MetricsCollector();
    ~MetricsCollector();
    
    // Start collecting metrics
    void setLogger(Logger* logger);
    void setModelName(const std::string& model_name);
    void setWindowsProcessed(uint64_t windows);
    
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
    Logger* external_logger_;
    Logger* active_logger_;
    bool owns_logger_;
    std::atomic<uint64_t> windows_processed_;
    std::string model_name_;
    std::mutex model_mutex_;
    
    void collectLoop();
};

#endif // METRICS_COLLECTOR_H

