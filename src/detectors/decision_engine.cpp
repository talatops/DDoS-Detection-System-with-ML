#include "decision_engine.h"
#include <algorithm>
#include <sstream>

DecisionEngine::DecisionEngine(double entropy_threshold, double ml_threshold,
                               double cusum_threshold, double pca_threshold,
                               bool use_weighted)
    : entropy_detector_(),
      cusum_detector_(cusum_threshold),
      pca_detector_(3, 1000, pca_threshold),
      entropy_threshold_(entropy_threshold),
      ml_threshold_(ml_threshold),
      cusum_threshold_(cusum_threshold),
      pca_threshold_(pca_threshold),
      use_weighted_(use_weighted),
      w_entropy_(0.4),
      w_ml_(0.4),
      w_cusum_(0.1),
      w_pca_(0.1) {
}

DecisionEngine::~DecisionEngine() {
}

void DecisionEngine::setWeights(double w_entropy, double w_ml, double w_cusum, double w_pca) {
    w_entropy_ = w_entropy;
    w_ml_ = w_ml;
    w_cusum_ = w_cusum;
    w_pca_ = w_pca;
    
    // Normalize weights
    double total = w_entropy_ + w_ml_ + w_cusum_ + w_pca_;
    if (total > 0) {
        w_entropy_ /= total;
        w_ml_ /= total;
        w_cusum_ /= total;
        w_pca_ /= total;
    }
}

void DecisionEngine::update(const EntropyDetector::EntropyFeatures& entropy_features,
                           double ml_probability,
                           const EntropyDetector::EntropyFeatures& current_features) {
    (void)ml_probability;  // May be used in future weighted detection
    (void)current_features;  // May be used for PCA updates
    // Update CUSUM with average entropy
    double avg_entropy = (entropy_features.src_ip_entropy + entropy_features.dst_ip_entropy +
                         entropy_features.src_port_entropy + entropy_features.dst_port_entropy +
                         entropy_features.packet_size_entropy + entropy_features.protocol_entropy) / 6.0;
    cusum_detector_.update(avg_entropy);
    
    // Update PCA detector
    pca_detector_.addTrainingSample(current_features);
}

DetectionResult DecisionEngine::detect(const EntropyDetector::EntropyFeatures& entropy_features,
                                      double ml_probability) {
    if (use_weighted_) {
        return detectWeighted(entropy_features, ml_probability);
    } else {
        return detectEnsemble(entropy_features, ml_probability);
    }
}

DetectionResult DecisionEngine::detectEnsemble(const EntropyDetector::EntropyFeatures& entropy_features,
                                               double ml_probability) {
    DetectionResult result;
    
    // Calculate scores
    result.entropy_score = entropy_detector_.getAnomalyScore(entropy_features);
    result.ml_score = ml_probability;
    result.cusum_score = cusum_detector_.getStatistic() / cusum_threshold_;  // Normalize
    result.pca_score = pca_detector_.getReconstructionError(entropy_features) / pca_threshold_;  // Normalize
    
    // Check each detector
    bool entropy_alert = result.entropy_score > entropy_threshold_;
    bool ml_alert = result.ml_score > ml_threshold_;
    bool cusum_alert = cusum_detector_.isAnomaly();
    bool pca_alert = pca_detector_.detectAnomaly(entropy_features);
    
    // Ensemble: Alert if ANY detector triggers
    result.is_attack = entropy_alert || ml_alert || cusum_alert || pca_alert;
    
    // Build trigger string
    std::vector<std::string> triggers;
    if (entropy_alert) triggers.push_back("Entropy");
    if (ml_alert) triggers.push_back("ML");
    if (cusum_alert) triggers.push_back("CUSUM");
    if (pca_alert) triggers.push_back("PCA");
    
    std::ostringstream oss;
    for (size_t i = 0; i < triggers.size(); ++i) {
        if (i > 0) oss << "+";
        oss << triggers[i];
    }
    result.detector_triggered = oss.str();
    
    // Combined score (max of all scores)
    result.combined_score = std::max({result.entropy_score, result.ml_score,
                                     result.cusum_score, result.pca_score});
    
    return result;
}

DetectionResult DecisionEngine::detectWeighted(const EntropyDetector::EntropyFeatures& entropy_features,
                                               double ml_probability) {
    DetectionResult result;
    
    // Calculate normalized scores
    result.entropy_score = entropy_detector_.getAnomalyScore(entropy_features);
    result.ml_score = ml_probability;
    result.cusum_score = std::min(1.0, cusum_detector_.getStatistic() / cusum_threshold_);
    result.pca_score = std::min(1.0, pca_detector_.getReconstructionError(entropy_features) / pca_threshold_);
    
    // Weighted combination
    result.combined_score = w_entropy_ * result.entropy_score +
                           w_ml_ * result.ml_score +
                           w_cusum_ * result.cusum_score +
                           w_pca_ * result.pca_score;
    
    // Threshold decision
    result.is_attack = result.combined_score > 0.5;  // Combined threshold
    
    // Determine which detectors contributed most
    std::vector<std::pair<double, std::string>> contributions = {
        {w_entropy_ * result.entropy_score, "Entropy"},
        {w_ml_ * result.ml_score, "ML"},
        {w_cusum_ * result.cusum_score, "CUSUM"},
        {w_pca_ * result.pca_score, "PCA"}
    };
    
    std::sort(contributions.rbegin(), contributions.rend());
    
    std::ostringstream oss;
    for (size_t i = 0; i < contributions.size() && contributions[i].first > 0.1; ++i) {
        if (i > 0) oss << "+";
        oss << contributions[i].second;
    }
    result.detector_triggered = oss.str();
    
    return result;
}

