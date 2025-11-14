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
    
    // Open log files
    alerts_file_.open(log_dir + "/alerts.csv");
    metrics_file_.open(log_dir + "/metrics.csv");
    blocking_file_.open(log_dir + "/blocking.csv");
    kernel_times_file_.open(log_dir + "/kernel_times.csv");
    
    if (!alerts_file_.is_open() || !metrics_file_.is_open() ||
        !blocking_file_.is_open() || !kernel_times_file_.is_open()) {
        std::cerr << "Failed to open log files" << std::endl;
        return false;
    }
    
    // Write headers
    alerts_file_ << "timestamp_ms,window_start_ms,src_ip,score,detector\n";
    metrics_file_ << "timestamp_ms,cpu_percent,gpu_percent,memory_mb,pps_in,pps_processed\n";
    blocking_file_ << "timestamp_ms,ip,impacted_packets,dropped_packets\n";
    kernel_times_file_ << "timestamp_ms,kernel_name,execution_time_ms\n";
    
    initialized_ = true;
    return true;
}

std::string Logger::ipToString(uint32_t ip) const {
    struct in_addr addr;
    addr.s_addr = ip;
    return std::string(inet_ntoa(addr));
}

void Logger::logAlert(uint64_t timestamp_ms, uint64_t window_start_ms,
                     uint32_t src_ip, double score, const std::string& detector) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    alerts_file_ << timestamp_ms << ","
                << window_start_ms << ","
                << ipToString(src_ip) << ","
                << score << ","
                << detector << "\n";
    alerts_file_.flush();
}

void Logger::logMetrics(uint64_t timestamp_ms, double cpu_percent, double gpu_percent,
                       double memory_mb, double pps_in, double pps_processed) {
    if (!initialized_) return;
    
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_file_ << timestamp_ms << ","
                 << cpu_percent << ","
                 << gpu_percent << ","
                 << memory_mb << ","
                 << pps_in << ","
                 << pps_processed << "\n";
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

