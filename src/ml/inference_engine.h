#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <string>
#include <vector>

// Forward declaration for Python object
struct _object;
typedef struct _object PyObject;

class MLInferenceEngine {
public:
    MLInferenceEngine();
    ~MLInferenceEngine();
    
    // Load model from joblib file with optional preprocessor/import hints
    bool loadModel(const std::string& model_path,
                   const std::string& preprocessor_path = "",
                   const std::vector<std::string>& module_imports = {});
    
    // Predict single sample (returns probability of attack class)
    double predict(const std::vector<double>& features);
    
    // Predict batch of samples (returns probabilities of attack class)
    std::vector<double> predictBatch(const std::vector<std::vector<double>>& features_batch);
    
    // Check if model is loaded
    bool isModelLoaded() const { return model_loaded_; }

private:
    bool model_loaded_;
    std::string model_path_;
    std::string preprocessor_path_;
    PyObject* model_;  // Python model object (RandomForestClassifier)
    PyObject* preprocessor_;  // Python preprocessor object (optional)
    
    // Apply preprocessing if available
    PyObject* applyPreprocessing(PyObject* features_array);
};

#endif // INFERENCE_ENGINE_H
