#ifndef FEATURE_BUILDER_H
#define FEATURE_BUILDER_H

#include <vector>
#include <cstdint>
#include "ingest/window_manager.h"
#include "detectors/entropy_cpu.h"

/**
 * FeatureBuilder converts GPU entropy results and window statistics
 * to ML feature vectors matching the training feature order.
 */
class FeatureBuilder {
public:
    FeatureBuilder();
    ~FeatureBuilder();
    
    /**
     * Build ML feature vector from GPU entropy results and window stats.
     * 
     * @param window Window statistics
     * @param gpu_entropy_results GPU entropy results (6 values: src_ip, dst_ip, src_port, dst_port, packet_size, protocol)
     * @param feature_vector Output feature vector (16 features matching training order)
     * @return true if successful
     */
    bool buildFeatures(const WindowStats& window,
                      const std::vector<double>& gpu_entropy_results,
                      std::vector<double>& feature_vector);
    
    /**
     * Build ML feature vector from CPU entropy features and window stats.
     * 
     * @param window Window statistics
     * @param entropy_features CPU entropy features
     * @param feature_vector Output feature vector (16 features matching training order)
     * @return true if successful
     */
    bool buildFeaturesFromCPU(const WindowStats& window,
                             const EntropyDetector::EntropyFeatures& entropy_features,
                             std::vector<double>& feature_vector);
    
    /**
     * Calculate top-N fraction from a counts map.
     */
    static double calculateTopNFraction(const std::unordered_map<uint32_t, uint32_t>& counts,
                                       uint32_t total, size_t n = 10);
    
    static double calculateTopNFraction(const std::unordered_map<uint16_t, uint32_t>& counts,
                                       uint32_t total, size_t n = 10);

private:
    /**
     * Calculate unique counts from window stats.
     */
    void calculateUniqueCounts(const WindowStats& window,
                             uint32_t& unique_src_ips,
                             uint32_t& unique_dst_ips,
                             uint32_t& unique_src_ports,
                             uint32_t& unique_dst_ports);
    
    /**
     * Calculate average packet size.
     */
    double calculateAvgPacketSize(const WindowStats& window);
};

#endif // FEATURE_BUILDER_H

