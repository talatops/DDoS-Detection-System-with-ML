#ifndef CUSUM_DETECTOR_H
#define CUSUM_DETECTOR_H

#include <cstdint>
#include <deque>
#include <cmath>
#include "detectors/entropy_cpu.h"

class CUSUMDetector {
public:
    CUSUMDetector(double threshold = 5.0, double drift = 0.5, size_t window_size = 100);
    ~CUSUMDetector();
    
    // Update detector with new entropy value
    void update(double entropy_value);
    
    // Check if anomaly detected
    bool isAnomaly() const { return anomaly_detected_; }
    
    // Get current CUSUM statistic
    double getStatistic() const { return s_positive_ + s_negative_; }
    
    // Reset detector
    void reset();
    
    // Set threshold
    void setThreshold(double threshold) { threshold_ = threshold; }
    
    // Get baseline mean (for debugging)
    double getBaselineMean() const { return baseline_mean_; }

private:
    double threshold_;
    double drift_;
    size_t window_size_;
    
    double s_positive_;  // Positive cumulative sum
    double s_negative_;   // Negative cumulative sum
    double baseline_mean_;
    bool anomaly_detected_;
    
    std::deque<double> recent_values_;  // For baseline calculation
    
    void updateBaseline(double value);
    void updateCUSUM(double value);
};

#endif // CUSUM_DETECTOR_H

