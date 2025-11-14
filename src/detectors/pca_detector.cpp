#include "pca_detector.h"
#include <algorithm>
#include <numeric>
#include <cmath>

PCADetector::PCADetector(size_t n_components, size_t training_size, double threshold)
    : n_components_(n_components), training_size_(training_size),
      threshold_(threshold), trained_(false) {
    mean_.resize(6, 0.0);  // 6 entropy features
}

PCADetector::~PCADetector() {
}

std::vector<double> PCADetector::featuresToVector(const EntropyDetector::EntropyFeatures& features) {
    return {
        features.src_ip_entropy,
        features.dst_ip_entropy,
        features.src_port_entropy,
        features.dst_port_entropy,
        features.packet_size_entropy,
        features.protocol_entropy
    };
}

void PCADetector::addTrainingSample(const EntropyDetector::EntropyFeatures& features) {
    std::vector<double> vec = featuresToVector(features);
    training_data_.push_back(vec);
    
    if (training_data_.size() > training_size_) {
        training_data_.pop_front();
    }
    
    // Recompute PCA if we have enough samples
    if (training_data_.size() >= training_size_ && !trained_) {
        computePCA();
        trained_ = true;
    }
}

void PCADetector::computePCA() {
    if (training_data_.size() < 2) return;
    
    size_t n_features = 6;  // Number of entropy features
    size_t n_samples = training_data_.size();
    
    // Calculate mean
    mean_.assign(n_features, 0.0);
    for (const auto& sample : training_data_) {
        for (size_t i = 0; i < n_features; ++i) {
            mean_[i] += sample[i];
        }
    }
    for (size_t i = 0; i < n_features; ++i) {
        mean_[i] /= n_samples;
    }
    
    // Center the data
    std::vector<std::vector<double>> centered_data;
    for (const auto& sample : training_data_) {
        std::vector<double> centered(n_features);
        for (size_t i = 0; i < n_features; ++i) {
            centered[i] = sample[i] - mean_[i];
        }
        centered_data.push_back(centered);
    }
    
    // Compute covariance matrix (simplified - using first n_components principal directions)
    // For simplicity, we'll use the first n_components dimensions as principal components
    // In a full implementation, we'd compute eigenvectors of covariance matrix
    principal_components_.clear();
    for (size_t i = 0; i < n_components_ && i < n_features; ++i) {
        std::vector<double> component(n_features, 0.0);
        component[i] = 1.0;  // Simplified: use unit vectors
        principal_components_.push_back(component);
    }
}

std::vector<double> PCADetector::project(const std::vector<double>& data) {
    std::vector<double> projected(n_components_, 0.0);
    
    // Center data
    std::vector<double> centered(data.size());
    for (size_t i = 0; i < data.size(); ++i) {
        centered[i] = data[i] - mean_[i];
    }
    
    // Project onto principal components
    for (size_t i = 0; i < n_components_; ++i) {
        double dot = 0.0;
        for (size_t j = 0; j < centered.size(); ++j) {
            dot += centered[j] * principal_components_[i][j];
        }
        projected[i] = dot;
    }
    
    return projected;
}

std::vector<double> PCADetector::reconstruct(const std::vector<double>& projected) {
    std::vector<double> reconstructed(6, 0.0);  // 6 features
    
    // Reconstruct from principal components
    for (size_t i = 0; i < n_components_; ++i) {
        for (size_t j = 0; j < reconstructed.size(); ++j) {
            reconstructed[j] += projected[i] * principal_components_[i][j];
        }
    }
    
    // Add mean back
    for (size_t i = 0; i < reconstructed.size(); ++i) {
        reconstructed[i] += mean_[i];
    }
    
    return reconstructed;
}

double PCADetector::computeError(const std::vector<double>& original, const std::vector<double>& reconstructed) {
    double error = 0.0;
    for (size_t i = 0; i < original.size(); ++i) {
        double diff = original[i] - reconstructed[i];
        error += diff * diff;
    }
    return std::sqrt(error);
}

bool PCADetector::detectAnomaly(const EntropyDetector::EntropyFeatures& features) {
    if (!trained_) {
        // If not trained, add to training data
        addTrainingSample(features);
        return false;
    }
    
    double error = getReconstructionError(features);
    return error > threshold_;
}

double PCADetector::getReconstructionError(const EntropyDetector::EntropyFeatures& features) {
    if (!trained_) return 0.0;
    
    std::vector<double> original = featuresToVector(features);
    std::vector<double> projected = project(original);
    std::vector<double> reconstructed = reconstruct(projected);
    
    return computeError(original, reconstructed);
}

void PCADetector::reset() {
    training_data_.clear();
    principal_components_.clear();
    mean_.assign(6, 0.0);
    trained_ = false;
}

