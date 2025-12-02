#include "utils/config_loader.h"
#include "utils/simple_json.h"
#include <fstream>
#include <iostream>
#include <sstream>

namespace {

bool readFile(const std::string& path, std::string& content) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Failed to open config file: " << path << std::endl;
        return false;
    }
    std::ostringstream oss;
    oss << file.rdbuf();
    content = oss.str();
    return true;
}

std::string getString(const SimpleJson::Value& value, const std::string& key,
                      const std::string& fallback = "") {
    if (!value.isObject()) return fallback;
    const auto* node = value.find(key);
    if (node && node->isString()) {
        return node->string_value;
    }
    return fallback;
}

double getNumber(const SimpleJson::Value& value, const std::string& key,
                 double fallback = 0.0) {
    if (!value.isObject()) return fallback;
    const auto* node = value.find(key);
    if (node && node->isNumber()) {
        return node->number_value;
    }
    return fallback;
}

bool getBool(const SimpleJson::Value& value, const std::string& key, bool fallback) {
    if (!value.isObject()) return fallback;
    const auto* node = value.find(key);
    if (!node) return fallback;
    if (node->isBool()) return node->bool_value;
    if (node->isNumber()) return node->number_value != 0.0;
    return fallback;
}

std::vector<std::string> getStringArray(const SimpleJson::Value& value,
                                        const std::string& key) {
    std::vector<std::string> result;
    if (!value.isObject()) return result;
    const auto* node = value.find(key);
    if (!node || !node->isArray()) return result;
    for (const auto& entry : node->array_values) {
        if (entry.isString()) {
            result.push_back(entry.string_value);
        }
    }
    return result;
}

}  // namespace

bool loadDetectionConfig(const std::string& path, DetectionConfig& config) {
    std::string content;
    if (!readFile(path, content)) {
        return false;
    }
    SimpleJson::Value root;
    std::string error;
    if (!SimpleJson::parse(content, root, error)) {
        std::cerr << "Failed to parse detection config: " << error << std::endl;
        return false;
    }
    config.entropy_threshold = getNumber(root, "entropy_threshold", config.entropy_threshold);
    config.ml_threshold = getNumber(root, "ml_threshold", config.ml_threshold);
    config.cusum_threshold = getNumber(root, "cusum_threshold", config.cusum_threshold);
    config.pca_threshold = getNumber(root, "pca_threshold", config.pca_threshold);
    config.use_weighted = getBool(root, "use_weighted", config.use_weighted);

    const SimpleJson::Value* weights = root.find("weights");
    if (weights && weights->isObject()) {
        config.w_entropy = getNumber(*weights, "entropy", config.w_entropy);
        config.w_ml = getNumber(*weights, "ml", config.w_ml);
        config.w_cusum = getNumber(*weights, "cusum", config.w_cusum);
        config.w_pca = getNumber(*weights, "pca", config.w_pca);
    }
    return true;
}

bool loadModelRegistry(const std::string& path, MLModelRegistry& registry) {
    std::string content;
    if (!readFile(path, content)) {
        return false;
    }
    SimpleJson::Value root;
    std::string error;
    if (!SimpleJson::parse(content, root, error)) {
        std::cerr << "Failed to parse model registry: " << error << std::endl;
        return false;
    }
    registry.default_model = getString(root, "default_model", "rf");
    registry.selected_model = getString(root, "selected_model", registry.default_model);
    const SimpleJson::Value* models = root.find("models");
    if (!models || !models->isArray()) {
        std::cerr << "Model registry missing models array" << std::endl;
        return false;
    }
    registry.models.clear();
    for (const auto& item : models->array_values) {
        if (!item.isObject()) continue;
        ModelInfo info;
        info.name = getString(item, "name", "");
        info.type = getString(item, "type", "");
        info.path = getString(item, "path", "");
        info.preprocessor = getString(item, "preprocessor", "");
        info.imports = getStringArray(item, "imports");
        info.recall = getNumber(item, "recall", 0.0);
        info.fpr = getNumber(item, "false_positive_rate", 0.0);
        if (!info.name.empty() && !info.path.empty()) {
            registry.models.push_back(info);
        }
    }
    return !registry.models.empty();
}

const ModelInfo* resolveModel(const MLModelRegistry& registry,
                              const std::string& requested_name,
                              std::string& resolved_name) {
    std::string target = requested_name;
    if (target == "auto" || target.empty()) {
        target = registry.selected_model.empty() ? registry.default_model : registry.selected_model;
    }
    for (const auto& model : registry.models) {
        if (model.name == target) {
            resolved_name = model.name;
            return &model;
        }
    }
    if (!registry.models.empty()) {
        resolved_name = registry.models.front().name;
        return &registry.models.front();
    }
    resolved_name.clear();
    return nullptr;
}

