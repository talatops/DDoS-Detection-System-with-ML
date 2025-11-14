#ifndef ENTROPY_CPU_H
#define ENTROPY_CPU_H

#include <cstdint>
#include <unordered_map>
#include <cmath>
#include "ingest/window_manager.h"

class EntropyDetector {
public:
    EntropyDetector();
    ~EntropyDetector();
    
    // Calculate entropy from a histogram
    static double calculateEntropy(const std::unordered_map<uint32_t, uint32_t>& counts, uint32_t total);
    static double calculateEntropy(const std::unordered_map<uint16_t, uint32_t>& counts, uint32_t total);
    static double calculateEntropy(const std::unordered_map<uint8_t, uint32_t>& counts, uint32_t total);
    
    // Calculate all entropy features for a window
    struct EntropyFeatures {
        double src_ip_entropy;
        double dst_ip_entropy;
        double src_port_entropy;
        double dst_port_entropy;
        double packet_size_entropy;
        double protocol_entropy;
        
        EntropyFeatures() : src_ip_entropy(0), dst_ip_entropy(0),
                           src_port_entropy(0), dst_port_entropy(0),
                           packet_size_entropy(0), protocol_entropy(0) {}
    };
    
    EntropyFeatures calculateFeatures(const WindowStats& window);
    
    // Detect anomaly based on entropy threshold
    bool isAnomaly(const EntropyFeatures& features, double threshold = 0.5);
    
    // Get normalized entropy score (0-1)
    double getAnomalyScore(const EntropyFeatures& features);
};

#endif // ENTROPY_CPU_H

