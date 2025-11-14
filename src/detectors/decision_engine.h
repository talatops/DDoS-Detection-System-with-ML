#ifndef DECISION_ENGINE_H
#define DECISION_ENGINE_H

#include <cstdint>
#include <string>
#include <vector>
#include "detectors/entropy_cpu.h"
#include "detectors/cusum_detector.h"
#include "detectors/pca_detector.h"

struct DetectionResult {
    bool is_attack;
    double combined_score;
    double entropy_score;
    double ml_score;
    double cusum_score;
    double pca_score;
    std::string detector_triggered;  // Which detector(s) triggered
    
    DetectionResult() : is_attack(false), combined_score(0.0),
                       entropy_score(0.0), ml_score(0.0),
                       cusum_score(0.0), pca_score(0.0) {}
};

class DecisionEngine {
public:
    DecisionEngine(double entropy_threshold = 0.5,
                   double ml_threshold = 0.5,
                   double cusum_threshold = 5.0,
                   double pca_threshold = 0.1,
                   bool use_weighted = false);
    ~DecisionEngine();
    
    // Update detectors with new window data
    void update(const EntropyDetector::EntropyFeatures& entropy_features,
                double ml_probability,
                const EntropyDetector::EntropyFeatures& current_features);
    
    // Make detection decision
    DetectionResult detect(const EntropyDetector::EntropyFeatures& entropy_features,
                          double ml_probability);
    
    // Set thresholds
    void setEntropyThreshold(double threshold) { entropy_threshold_ = threshold; }
    void setMLThreshold(double threshold) { ml_threshold_ = threshold; }
    void setCUSUMThreshold(double threshold) { cusum_threshold_ = threshold; }
    void setPCAThreshold(double threshold) { pca_threshold_ = threshold; }
    
    // Set weights for weighted scoring
    void setWeights(double w_entropy, double w_ml, double w_cusum, double w_pca);
    
    // Get detector instances (for external updates)
    CUSUMDetector& getCUSUMDetector() { return cusum_detector_; }
    PCADetector& getPCADetector() { return pca_detector_; }

private:
    EntropyDetector entropy_detector_;
    CUSUMDetector cusum_detector_;
    PCADetector pca_detector_;
    
    double entropy_threshold_;
    double ml_threshold_;
    double cusum_threshold_;
    double pca_threshold_;
    
    bool use_weighted_;
    double w_entropy_;
    double w_ml_;
    double w_cusum_;
    double w_pca_;
    
    DetectionResult detectEnsemble(const EntropyDetector::EntropyFeatures& entropy_features,
                                   double ml_probability);
    DetectionResult detectWeighted(const EntropyDetector::EntropyFeatures& entropy_features,
                                   double ml_probability);
};

#endif // DECISION_ENGINE_H

