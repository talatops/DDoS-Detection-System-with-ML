#include "utils/metrics_collector.h"
#include "utils/resource_monitor.h"
#include "utils/logger.h"
#include <thread>
#include <chrono>
#include <cstdint>

MetricsCollector::MetricsCollector()
    : running_(false),
      collector_thread_(nullptr),
      external_logger_(nullptr),
      active_logger_(nullptr),
      owns_logger_(false),
      windows_processed_(0) {
}
void MetricsCollector::setModelName(const std::string& model_name) {
    std::lock_guard<std::mutex> lock(model_mutex_);
    model_name_ = model_name;
}

void MetricsCollector::setWindowsProcessed(uint64_t windows) {
    windows_processed_.store(windows);
}


MetricsCollector::~MetricsCollector() {
    stop();
}

void MetricsCollector::setLogger(Logger* logger) {
    external_logger_ = logger;
}

bool MetricsCollector::start(const std::string& output_dir) {
    if (running_) return false;
    
    if (external_logger_) {
        active_logger_ = external_logger_;
        owns_logger_ = false;
    } else {
        if (!logger_.initialize(output_dir)) {
            return false;
        }
        active_logger_ = &logger_;
        owns_logger_ = true;
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
    if (owns_logger_ && active_logger_) {
        active_logger_->close();
    }
    active_logger_ = nullptr;
}

void MetricsCollector::collectLoop() {
    while (running_) {
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        
        ResourceMetrics metrics = monitor_.getMetrics();
        
        if (active_logger_) {
            std::string model_name_copy;
            {
                std::lock_guard<std::mutex> lock(model_mutex_);
                model_name_copy = model_name_;
            }
            active_logger_->logMetrics(now, metrics.cpu_percent, metrics.gpu_percent,
                                      metrics.memory_mb, metrics.pps_in, metrics.pps_processed,
                                      windows_processed_.load(), model_name_copy);
        }
        
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}

void MetricsCollector::updatePacketCount(uint64_t packets_in, uint64_t packets_processed) {
    monitor_.updatePacketCount(packets_in, packets_processed);
}

