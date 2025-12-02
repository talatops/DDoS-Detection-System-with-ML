#ifndef CONFIG_LOADER_H
#define CONFIG_LOADER_H

#include <string>
#include <vector>

struct DetectionConfig {
    double entropy_threshold = 0.7;
    double ml_threshold = 0.5;
    double cusum_threshold = 3.0;
    double pca_threshold = 2.0;
    bool use_weighted = false;
    double w_entropy = 0.4;
    double w_ml = 0.4;
    double w_cusum = 0.1;
    double w_pca = 0.1;
};

struct ModelInfo {
    std::string name;
    std::string type;
    std::string path;
    std::string preprocessor;
    std::vector<std::string> imports;
    double recall = 0.0;
    double fpr = 0.0;
};

struct MLModelRegistry {
    std::string default_model;
    std::string selected_model;
    std::vector<ModelInfo> models;
};

bool loadDetectionConfig(const std::string& path, DetectionConfig& config);
bool loadModelRegistry(const std::string& path, MLModelRegistry& registry);
const ModelInfo* resolveModel(const MLModelRegistry& registry,
                              const std::string& requested_name,
                              std::string& resolved_name);

#endif  // CONFIG_LOADER_H

