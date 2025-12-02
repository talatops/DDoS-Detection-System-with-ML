#include "logger.h"
#include <iostream>
#include <sstream>
#include <sys/stat.h>
#include <arpa/inet.h>

Logger::Logger() : initialized_(false) {
}

Logger::~Logger() {
    close();
}

bool Logger::initialize(const std::string& log_dir) {
    log_dir_ = log_dir;
    
    // Create directory if it doesn't exist
    struct stat info;
    if (stat(log_dir.c_str(), &info) != 0) {
        // Create directory (simplified - in production use filesystem library)
        std::string cmd = "mkdir -p " + log_dir;
        system(cmd.c_str());
    }
    
    // Open log files in append mode
    alerts_file_.open(log_dir + "/alerts.csv", std::ios::app);
    metrics_file_.open(log_dir + "/metrics.csv", std::ios::app);
    blocking_file_.open(log_dir + "/blocking.csv", std::ios::app);
    kernel_times_file_.open(log_dir + "/kernel_times.csv", std::ios::app);
    
    if (!alerts_file_.is_open() || !metrics_file_.is_open() ||
        !blocking_file_.is_open() || !kernel_times_file_.is_open()) {
        std::cerr << "Failed to open log files" << std::endl;
        return false;
    }
    
    // Write headers only if file is new/empty (check file position)
    std::streampos alerts_pos = alerts_file_.tellp();
    std::streampos metrics_pos = metrics_file_.tellp();
    std::streampos blocking_pos = blocking_file_.tellp();
    std::streampos kernel_pos = kernel_times_file_.tellp();
    
    if (alerts_pos == 0) {
        alerts_file_ << "timestamp_ms,window_start_ms,window_index,top_src_ip,"
                     << "entropy_score,ml_score,cusum_score,pca_score,combined_score,"
                     << "detector,model\n";
    }
    if (metrics_pos == 0) {
        metrics_file_ << "timestamp_ms,cpu_percent,gpu_percent,memory_mb,pps_in,pps_processed,"
                      << "windows_processed,model\n";
    }
    if (blocking_pos == 0) {
        blocking_file_ << "timestamp_ms,ip,impacted_packets,dropped_packets\n";
    }
    if (kernel_pos == 0) {
        kernel_times_file_ << "timestamp_ms,kernel_name,execution_time_ms\n";
    }
    
    initialized_ = true;
    return true;
}

std::string Logger::ipToString(uint32_t ip) const {
    struct in_addr addr;
    addr.s_addr = ip;
    return std::string(inet_ntoa(addr));
}

void Logger::logAlert(uint64_t timestamp_ms, uint64_t window_start_ms,
                     uint64_t window_index, uint32_t top_src_ip,
                     double entropy_score, double ml_score,
                     double cusum_score, double pca_score,
                     double combined_score, const std::string& detector,
                     const std::string& model_name) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    alerts_file_ << timestamp_ms << ","
                 << window_start_ms << ","
                 << window_index << ","
                 << ipToString(top_src_ip) << ","
                 << entropy_score << ","
                 << ml_score << ","
                 << cusum_score << ","
                 << pca_score << ","
                 << combined_score << ","
                 << detector << ","
                 << model_name << "\n";
    alerts_file_.flush();
}

void Logger::logMetrics(uint64_t timestamp_ms, double cpu_percent, double gpu_percent,
                       double memory_mb, double pps_in, double pps_processed,
                       uint64_t windows_processed, const std::string& model_name) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_file_ << timestamp_ms << ","
                 << cpu_percent << ","
                 << gpu_percent << ","
                 << memory_mb << ","
                 << pps_in << ","
                 << pps_processed << ","
                 << windows_processed << ","
                 << model_name << "\n";
    metrics_file_.flush();
}

void Logger::logBlocking(uint64_t timestamp_ms, uint32_t ip, uint64_t impacted_packets,
                        uint64_t dropped_packets) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    blocking_file_ << timestamp_ms << ","
                   << ipToString(ip) << ","
                   << impacted_packets << ","
                   << dropped_packets << "\n";
    blocking_file_.flush();
}

void Logger::logKernelTime(uint64_t timestamp_ms, const std::string& kernel_name,
                          double execution_time_ms) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    kernel_times_file_ << timestamp_ms << ","
                       << kernel_name << ","
                       << execution_time_ms << "\n";
    kernel_times_file_.flush();
}

void Logger::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (alerts_file_.is_open()) alerts_file_.close();
    if (metrics_file_.is_open()) metrics_file_.close();
    if (blocking_file_.is_open()) blocking_file_.close();
    if (kernel_times_file_.is_open()) kernel_times_file_.close();
    initialized_ = false;
}

