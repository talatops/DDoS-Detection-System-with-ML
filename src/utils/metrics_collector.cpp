#include "utils/metrics_collector.h"
#include "utils/resource_monitor.h"
#include "utils/logger.h"
#include <thread>
#include <chrono>
#include <cstdint>

MetricsCollector::MetricsCollector() : running_(false), collector_thread_(nullptr) {
}

MetricsCollector::~MetricsCollector() {
    stop();
}

bool MetricsCollector::start(const std::string& output_dir) {
    if (running_) return false;
    
    if (!logger_.initialize(output_dir)) {
        return false;
    }
    
    running_ = true;
    collector_thread_ = new std::thread(&MetricsCollector::collectLoop, this);
    return true;
}

void MetricsCollector::stop() {
    if (!running_) return;
    
    running_ = false;
    if (collector_thread_) {
        collector_thread_->join();
        delete collector_thread_;
        collector_thread_ = nullptr;
    }
    logger_.close();
}

void MetricsCollector::collectLoop() {
    while (running_) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        ResourceMetrics metrics = monitor_.getMetrics();
        
        logger_.logMetrics(now, metrics.cpu_percent, metrics.gpu_percent,
                          metrics.memory_mb, metrics.pps_in, metrics.pps_processed);
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void MetricsCollector::updatePacketCount(uint64_t packets_in, uint64_t packets_processed) {
    monitor_.updatePacketCount(packets_in, packets_processed);
}

