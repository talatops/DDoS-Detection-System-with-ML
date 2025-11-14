#include "cusum_detector.h"
#include <algorithm>
#include <numeric>

CUSUMDetector::CUSUMDetector(double threshold, double drift, size_t window_size)
    : threshold_(threshold), drift_(drift), window_size_(window_size),
      s_positive_(0.0), s_negative_(0.0), baseline_mean_(0.0), anomaly_detected_(false) {
}

CUSUMDetector::~CUSUMDetector() {
}

void CUSUMDetector::updateBaseline(double value) {
    recent_values_.push_back(value);
    
    if (recent_values_.size() > window_size_) {
        recent_values_.pop_front();
    }
    
    // Calculate mean of recent values
    if (!recent_values_.empty()) {
        double sum = std::accumulate(recent_values_.begin(), recent_values_.end(), 0.0);
        baseline_mean_ = sum / recent_values_.size();
    } else {
        baseline_mean_ = value;
    }
}

void CUSUMDetector::updateCUSUM(double value) {
    // Calculate deviation from baseline
    double deviation = value - baseline_mean_;
    
    // Update positive CUSUM (for increases)
    s_positive_ = std::max(0.0, s_positive_ + deviation - drift_);
    
    // Update negative CUSUM (for decreases)
    s_negative_ = std::max(0.0, s_negative_ - deviation - drift_);
    
    // Check if threshold exceeded
    double total_stat = s_positive_ + s_negative_;
    anomaly_detected_ = (total_stat > threshold_);
}

void CUSUMDetector::update(double entropy_value) {
    updateBaseline(entropy_value);
    updateCUSUM(entropy_value);
}

void CUSUMDetector::reset() {
    s_positive_ = 0.0;
    s_negative_ = 0.0;
    baseline_mean_ = 0.0;
    anomaly_detected_ = false;
    recent_values_.clear();
}

