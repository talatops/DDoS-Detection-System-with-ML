#ifndef LOGGER_H
#define LOGGER_H

#include <string>
#include <fstream>
#include <mutex>
#include <cstdint>

class Logger {
public:
    Logger();
    ~Logger();
    
    // Initialize logger with output directory
    bool initialize(const std::string& log_dir = "logs");
    
    // Log alert
    void logAlert(uint64_t timestamp_ms, uint64_t window_start_ms,
                 uint32_t src_ip, double score, const std::string& detector);
    
    // Log metrics
    void logMetrics(uint64_t timestamp_ms, double cpu_percent, double gpu_percent,
                   double memory_mb, double pps_in, double pps_processed);
    
    // Log blocking action
    void logBlocking(uint64_t timestamp_ms, uint32_t ip, uint64_t impacted_packets,
                    uint64_t dropped_packets);
    
    // Log kernel execution time
    void logKernelTime(uint64_t timestamp_ms, const std::string& kernel_name,
                      double execution_time_ms);
    
    // Close log files
    void close();

private:
    std::string log_dir_;
    std::ofstream alerts_file_;
    std::ofstream metrics_file_;
    std::ofstream blocking_file_;
    std::ofstream kernel_times_file_;
    std::mutex mutex_;
    bool initialized_;
    
    std::string ipToString(uint32_t ip) const;
};

#endif // LOGGER_H

