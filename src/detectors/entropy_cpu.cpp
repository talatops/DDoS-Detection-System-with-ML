#include "entropy_cpu.h"
#include <algorithm>
#include <limits>

EntropyDetector::EntropyDetector() {
}

EntropyDetector::~EntropyDetector() {
}

double EntropyDetector::calculateEntropy(const std::unordered_map<uint32_t, uint32_t>& counts, uint32_t total) {
    if (total == 0) return 0.0;
    
    double entropy = 0.0;
    const double log2 = std::log(2.0);
    
    for (const auto& pair : counts) {
        if (pair.second > 0) {
            double p = static_cast<double>(pair.second) / static_cast<double>(total);
            entropy -= p * std::log(p) / log2;  // log2
        }
    }
    
    return entropy;
}

double EntropyDetector::calculateEntropy(const std::unordered_map<uint16_t, uint32_t>& counts, uint32_t total) {
    if (total == 0) return 0.0;
    
    double entropy = 0.0;
    const double log2 = std::log(2.0);
    
    for (const auto& pair : counts) {
        if (pair.second > 0) {
            double p = static_cast<double>(pair.second) / static_cast<double>(total);
            entropy -= p * std::log(p) / log2;  // log2
        }
    }
    
    return entropy;
}

double EntropyDetector::calculateEntropy(const std::unordered_map<uint8_t, uint32_t>& counts, uint32_t total) {
    if (total == 0) return 0.0;
    
    double entropy = 0.0;
    const double log2 = std::log(2.0);
    
    for (const auto& pair : counts) {
        if (pair.second > 0) {
            double p = static_cast<double>(pair.second) / static_cast<double>(total);
            entropy -= p * std::log(p) / log2;  // log2
        }
    }
    
    return entropy;
}

EntropyDetector::EntropyFeatures EntropyDetector::calculateFeatures(const WindowStats& window) {
    EntropyFeatures features;
    
    if (window.total_packets == 0) {
        return features;
    }
    
    features.src_ip_entropy = calculateEntropy(window.src_ip_counts, window.total_packets);
    features.dst_ip_entropy = calculateEntropy(window.dst_ip_counts, window.total_packets);
    features.src_port_entropy = calculateEntropy(window.src_port_counts, window.total_packets);
    features.dst_port_entropy = calculateEntropy(window.dst_port_counts, window.total_packets);
    features.packet_size_entropy = calculateEntropy(window.packet_size_counts, window.total_packets);
    features.protocol_entropy = calculateEntropy(window.protocol_counts, window.total_packets);
    
    return features;
}

bool EntropyDetector::isAnomaly(const EntropyFeatures& features, double threshold) {
    double score = getAnomalyScore(features);
    return score > threshold;
}

double EntropyDetector::getAnomalyScore(const EntropyFeatures& features) {
    // Normalize entropy values (max entropy for N items is log2(N))
    // For DDoS attacks, we expect:
    // - Low source IP entropy (many packets from few IPs)
    // - Low destination IP entropy (targeting one victim)
    // - Low port entropy (same ports)
    
    // Calculate average entropy (normalized)
    double avg_entropy = (features.src_ip_entropy + features.dst_ip_entropy + 
                         features.src_port_entropy + features.dst_port_entropy +
                         features.packet_size_entropy + features.protocol_entropy) / 6.0;
    
    // Normalize to 0-1 range (assuming max entropy around 10-15 bits)
    double normalized = std::min(1.0, avg_entropy / 15.0);
    
    // Lower entropy = higher anomaly score
    return 1.0 - normalized;
}

