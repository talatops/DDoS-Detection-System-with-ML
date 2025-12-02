#include "feature_builder.h"
#include "detectors/entropy_cpu.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>

FeatureBuilder::FeatureBuilder() {
}

FeatureBuilder::~FeatureBuilder() {
}

double FeatureBuilder::calculateTopNFraction(const std::unordered_map<uint32_t, uint32_t>& counts,
                                             uint32_t total, size_t n) {
    if (total == 0) return 1.0;
    
    std::vector<uint32_t> sorted_counts;
    for (const auto& pair : counts) {
        sorted_counts.push_back(pair.second);
    }
    std::sort(sorted_counts.rbegin(), sorted_counts.rend());
    
    size_t top_n = std::min(n, sorted_counts.size());
    uint32_t top_n_sum = 0;
    for (size_t i = 0; i < top_n; ++i) {
        top_n_sum += sorted_counts[i];
    }
    
    return static_cast<double>(top_n_sum) / static_cast<double>(total);
}

double FeatureBuilder::calculateTopNFraction(const std::unordered_map<uint16_t, uint32_t>& counts,
                                             uint32_t total, size_t n) {
    if (total == 0) return 1.0;
    
    std::vector<uint32_t> sorted_counts;
    for (const auto& pair : counts) {
        sorted_counts.push_back(pair.second);
    }
    std::sort(sorted_counts.rbegin(), sorted_counts.rend());
    
    size_t top_n = std::min(n, sorted_counts.size());
    uint32_t top_n_sum = 0;
    for (size_t i = 0; i < top_n; ++i) {
        top_n_sum += sorted_counts[i];
    }
    
    return static_cast<double>(top_n_sum) / static_cast<double>(total);
}

void FeatureBuilder::calculateUniqueCounts(const WindowStats& window,
                                          uint32_t& unique_src_ips,
                                          uint32_t& unique_dst_ips,
                                          uint32_t& unique_src_ports,
                                          uint32_t& unique_dst_ports) {
    unique_src_ips = static_cast<uint32_t>(window.src_ip_counts.size());
    unique_dst_ips = static_cast<uint32_t>(window.dst_ip_counts.size());
    unique_src_ports = static_cast<uint32_t>(window.src_port_counts.size());
    unique_dst_ports = static_cast<uint32_t>(window.dst_port_counts.size());
}

double FeatureBuilder::calculateAvgPacketSize(const WindowStats& window) {
    if (window.total_packets == 0) return 0.0;
    return static_cast<double>(window.total_bytes) / static_cast<double>(window.total_packets);
}

bool FeatureBuilder::buildFeatures(const WindowStats& window,
                                  const std::vector<double>& gpu_entropy_results,
                                  std::vector<double>& feature_vector) {
    // GPU entropy results should have 6 values per window
    if (gpu_entropy_results.size() < 6) {
        return false;
    }
    
    feature_vector.clear();
    feature_vector.reserve(24);
    
    // Feature order matching training script (24 features):
    // 1. total_packets
    feature_vector.push_back(static_cast<double>(window.total_packets));
    
    // 2. total_bytes
    feature_vector.push_back(static_cast<double>(window.total_bytes));
    
    // 3. flow_duration (window duration in microseconds, convert to milliseconds)
    uint64_t window_duration_us = window.window_end_us - window.window_start_us;
    double flow_duration_ms = static_cast<double>(window_duration_us) / 1000.0;
    feature_vector.push_back(flow_duration_ms);
    
    // 4. flow_bytes_per_sec
    double flow_bytes_per_sec = 0.0;
    if (window_duration_us > 0) {
        flow_bytes_per_sec = (static_cast<double>(window.total_bytes) * 1000000.0) / static_cast<double>(window_duration_us);
    }
    feature_vector.push_back(flow_bytes_per_sec);
    
    // 5. flow_packets_per_sec
    double flow_packets_per_sec = 0.0;
    if (window_duration_us > 0) {
        flow_packets_per_sec = (static_cast<double>(window.total_packets) * 1000000.0) / static_cast<double>(window_duration_us);
    }
    feature_vector.push_back(flow_packets_per_sec);
    
    // 6. avg_packet_size
    feature_vector.push_back(calculateAvgPacketSize(window));
    
    // 7-8. packet_size_mean and std (calculate from packet_size_counts)
    double packet_size_mean = 0.0;
    double packet_size_std = 0.0;
    if (!window.packet_size_counts.empty()) {
        double sum = 0.0;
        double sum_sq = 0.0;
        uint32_t count = 0;
        for (const auto& pair : window.packet_size_counts) {
            double size = static_cast<double>(pair.first);
            uint32_t freq = pair.second;
            sum += size * freq;
            sum_sq += size * size * freq;
            count += freq;
        }
        if (count > 0) {
            packet_size_mean = sum / count;
            double variance = (sum_sq / count) - (packet_size_mean * packet_size_mean);
            packet_size_std = std::sqrt(std::max(0.0, variance));
        }
    }
    feature_vector.push_back(packet_size_mean);
    feature_vector.push_back(packet_size_std);
    
    // 9-11. TCP flag counts from window stats
    feature_vector.push_back(static_cast<double>(window.syn_packets));
    feature_vector.push_back(static_cast<double>(window.fin_packets));
    feature_vector.push_back(static_cast<double>(window.rst_packets));
    
    // 12-16. Entropy features (from GPU results)
    // Order: src_ip, dst_ip, src_port, dst_port, protocol
    feature_vector.push_back(gpu_entropy_results[0]);  // src_ip_entropy
    feature_vector.push_back(gpu_entropy_results[1]);  // dst_ip_entropy
    feature_vector.push_back(gpu_entropy_results[2]);  // src_port_entropy
    feature_vector.push_back(gpu_entropy_results[3]);  // dst_port_entropy
    feature_vector.push_back(gpu_entropy_results[5]);  // protocol_entropy
    
    // 17-20. Unique counts
    uint32_t unique_src_ips, unique_dst_ips, unique_src_ports, unique_dst_ports;
    calculateUniqueCounts(window, unique_src_ips, unique_dst_ips, unique_src_ports, unique_dst_ports);
    feature_vector.push_back(static_cast<double>(unique_src_ips));
    feature_vector.push_back(static_cast<double>(unique_dst_ips));
    feature_vector.push_back(static_cast<double>(unique_src_ports));
    feature_vector.push_back(static_cast<double>(unique_dst_ports));
    
    // 21-22. Top-N fractions
    double top10_src_fraction = calculateTopNFraction(window.src_ip_counts, window.total_packets, 10);
    double top10_dst_fraction = calculateTopNFraction(window.dst_ip_counts, window.total_packets, 10);
    feature_vector.push_back(top10_src_fraction);
    feature_vector.push_back(top10_dst_fraction);
    
    // 23. packet_size_entropy (coefficient of variation)
    double packet_size_entropy = 0.0;
    if (packet_size_mean > 0.0) {
        packet_size_entropy = packet_size_std / (packet_size_mean + 1e-6);
    }
    feature_vector.push_back(packet_size_entropy);
    
    // 24. flow_count
    feature_vector.push_back(static_cast<double>(window.flow_count));
    
    return feature_vector.size() == 24;
}

bool FeatureBuilder::buildFeaturesFromCPU(const WindowStats& window,
                                         const EntropyDetector::EntropyFeatures& entropy_features,
                                         std::vector<double>& feature_vector) {
    feature_vector.clear();
    feature_vector.reserve(24);
    
    // Feature order matching training script (24 features):
    // 1. total_packets
    feature_vector.push_back(static_cast<double>(window.total_packets));
    
    // 2. total_bytes
    feature_vector.push_back(static_cast<double>(window.total_bytes));
    
    // 3. flow_duration (window duration in microseconds, convert to milliseconds)
    uint64_t window_duration_us = window.window_end_us - window.window_start_us;
    double flow_duration_ms = static_cast<double>(window_duration_us) / 1000.0;
    feature_vector.push_back(flow_duration_ms);
    
    // 4. flow_bytes_per_sec
    double flow_bytes_per_sec = 0.0;
    if (window_duration_us > 0) {
        flow_bytes_per_sec = (static_cast<double>(window.total_bytes) * 1000000.0) / static_cast<double>(window_duration_us);
    }
    feature_vector.push_back(flow_bytes_per_sec);
    
    // 5. flow_packets_per_sec
    double flow_packets_per_sec = 0.0;
    if (window_duration_us > 0) {
        flow_packets_per_sec = (static_cast<double>(window.total_packets) * 1000000.0) / static_cast<double>(window_duration_us);
    }
    feature_vector.push_back(flow_packets_per_sec);
    
    // 6. avg_packet_size
    feature_vector.push_back(calculateAvgPacketSize(window));
    
    // 7-8. packet_size_mean and std (calculate from packet_size_counts)
    double packet_size_mean = 0.0;
    double packet_size_std = 0.0;
    if (!window.packet_size_counts.empty()) {
        double sum = 0.0;
        double sum_sq = 0.0;
        uint32_t count = 0;
        for (const auto& pair : window.packet_size_counts) {
            double size = static_cast<double>(pair.first);
            uint32_t freq = pair.second;
            sum += size * freq;
            sum_sq += size * size * freq;
            count += freq;
        }
        if (count > 0) {
            packet_size_mean = sum / count;
            double variance = (sum_sq / count) - (packet_size_mean * packet_size_mean);
            packet_size_std = std::sqrt(std::max(0.0, variance));
        }
    }
    feature_vector.push_back(packet_size_mean);
    feature_vector.push_back(packet_size_std);
    
    // 9-11. TCP flags from window stats
    feature_vector.push_back(static_cast<double>(window.syn_packets));
    feature_vector.push_back(static_cast<double>(window.fin_packets));
    feature_vector.push_back(static_cast<double>(window.rst_packets));
    
    // 12-16. Entropy features (from CPU)
    feature_vector.push_back(entropy_features.src_ip_entropy);
    feature_vector.push_back(entropy_features.dst_ip_entropy);
    feature_vector.push_back(entropy_features.src_port_entropy);
    feature_vector.push_back(entropy_features.dst_port_entropy);
    feature_vector.push_back(entropy_features.protocol_entropy);
    
    // 17-20. Unique counts
    uint32_t unique_src_ips, unique_dst_ips, unique_src_ports, unique_dst_ports;
    calculateUniqueCounts(window, unique_src_ips, unique_dst_ips, unique_src_ports, unique_dst_ports);
    feature_vector.push_back(static_cast<double>(unique_src_ips));
    feature_vector.push_back(static_cast<double>(unique_dst_ips));
    feature_vector.push_back(static_cast<double>(unique_src_ports));
    feature_vector.push_back(static_cast<double>(unique_dst_ports));
    
    // 21-22. Top-N fractions
    double top10_src_fraction = calculateTopNFraction(window.src_ip_counts, window.total_packets, 10);
    double top10_dst_fraction = calculateTopNFraction(window.dst_ip_counts, window.total_packets, 10);
    feature_vector.push_back(top10_src_fraction);
    feature_vector.push_back(top10_dst_fraction);
    
    // 23. packet_size_entropy (coefficient of variation)
    double packet_size_entropy = 0.0;
    if (packet_size_mean > 0.0) {
        packet_size_entropy = packet_size_std / (packet_size_mean + 1e-6);
    }
    feature_vector.push_back(packet_size_entropy);
    
    // 24. flow_count
    feature_vector.push_back(static_cast<double>(window.flow_count));
    
    return feature_vector.size() == 24;
}

