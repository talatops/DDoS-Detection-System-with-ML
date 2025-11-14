#ifndef PCA_DETECTOR_H
#define PCA_DETECTOR_H

#include <cstdint>
#include <vector>
#include <deque>
#include <cmath>
#include "detectors/entropy_cpu.h"

class PCADetector {
public:
    PCADetector(size_t n_components = 3, size_t training_size = 1000, double threshold = 0.1);
    ~PCADetector();
    
    // Add training sample
    void addTrainingSample(const EntropyDetector::EntropyFeatures& features);
    
    // Check if training is complete
    bool isTrained() const { return trained_; }
    
    // Detect anomaly using PCA reconstruction error
    bool detectAnomaly(const EntropyDetector::EntropyFeatures& features);
    
    // Get reconstruction error
    double getReconstructionError(const EntropyDetector::EntropyFeatures& features);
    
    // Reset detector
    void reset();
    
    // Set threshold
    void setThreshold(double threshold) { threshold_ = threshold; }

private:
    size_t n_components_;
    size_t training_size_;
    double threshold_;
    bool trained_;
    
    std::deque<std::vector<double>> training_data_;
    std::vector<std::vector<double>> principal_components_;  // Eigenvectors
    std::vector<double> mean_;  // Mean of training data
    
    // Simple PCA implementation
    void computePCA();
    std::vector<double> project(const std::vector<double>& data);
    std::vector<double> reconstruct(const std::vector<double>& projected);
    double computeError(const std::vector<double>& original, const std::vector<double>& reconstructed);
    
    // Convert features to vector
    std::vector<double> featuresToVector(const EntropyDetector::EntropyFeatures& features);
};

#endif // PCA_DETECTOR_H

