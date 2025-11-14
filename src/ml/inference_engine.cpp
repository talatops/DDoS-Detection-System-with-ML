#include "ml/inference_engine.h"
#include <Python.h>
#include <numpy/arrayobject.h>
#include <iostream>
#include <vector>
#include <stdexcept>

MLInferenceEngine::MLInferenceEngine() : model_loaded_(false), model_(nullptr), preprocessor_(nullptr) {
    // Initialize Python interpreter (if not already initialized)
    if (!Py_IsInitialized()) {
        Py_Initialize();
        import_array();
    }
}

MLInferenceEngine::~MLInferenceEngine() {
    // Release Python objects
    if (model_ != nullptr) {
        Py_DECREF(model_);
        model_ = nullptr;
    }
    if (preprocessor_ != nullptr) {
        Py_DECREF(preprocessor_);
        preprocessor_ = nullptr;
    }
    
    // Note: We don't finalize Python here as it may be used elsewhere
    // Py_Finalize();
}

bool MLInferenceEngine::loadModel(const std::string& model_path) {
    if (model_loaded_) {
        std::cerr << "Model already loaded" << std::endl;
        return false;
    }
    
    try {
        // Import joblib
        PyObject* joblib_module = PyImport_ImportModule("joblib");
        if (!joblib_module) {
            PyErr_Print();
            std::cerr << "Failed to import joblib module" << std::endl;
            return false;
        }
        
        // Get joblib.load function
        PyObject* load_func = PyObject_GetAttrString(joblib_module, "load");
        if (!load_func || !PyCallable_Check(load_func)) {
            Py_DECREF(joblib_module);
            std::cerr << "Failed to get joblib.load function" << std::endl;
            return false;
        }
        
        // Load model
        PyObject* path_obj = PyUnicode_FromString(model_path.c_str());
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, path_obj);
        
        model_ = PyObject_CallObject(load_func, args);
        Py_DECREF(args);
        Py_DECREF(load_func);
        
        // Try to load preprocessor (optional)
        std::string preprocessor_path = model_path.substr(0, model_path.find_last_of('/')) + "/preprocessor.joblib";
        if (preprocessor_path.find('/') == std::string::npos) {
            preprocessor_path = "models/preprocessor.joblib";
        }
        
        path_obj = PyUnicode_FromString(preprocessor_path.c_str());
        args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, path_obj);
        
        preprocessor_ = PyObject_CallObject(load_func, args);
        Py_DECREF(args);
        
        if (preprocessor_) {
            Py_INCREF(preprocessor_);
            std::cout << "Preprocessor loaded from: " << preprocessor_path << std::endl;
        } else {
            PyErr_Clear();  // Clear error if preprocessor doesn't exist
            std::cout << "No preprocessor found, using raw features" << std::endl;
        }
        
        Py_DECREF(joblib_module);
        
        if (!model_) {
            PyErr_Print();
            std::cerr << "Failed to load model from: " << model_path << std::endl;
            return false;
        }
        
        // Increment reference count to keep model alive
        Py_INCREF(model_);
        
        model_path_ = model_path;
        model_loaded_ = true;
        
        std::cout << "Successfully loaded model from: " << model_path << std::endl;
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception loading model: " << e.what() << std::endl;
        return false;
    }
}

double MLInferenceEngine::predict(const std::vector<double>& features) {
    if (!model_loaded_ || !model_ || features.empty()) {
        return 0.0;
    }
    
    try {
        // Convert features to numpy array
        npy_intp dims[2] = {1, static_cast<npy_intp>(features.size())};
        PyObject* features_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, 
                                                             const_cast<double*>(features.data()));
        if (!features_array) {
            PyErr_Print();
            std::cerr << "Failed to create numpy array" << std::endl;
            return 0.0;
        }
        
        // Make array writeable (required for some operations)
        PyArray_ENABLEFLAGS((PyArrayObject*)features_array, NPY_ARRAY_WRITEABLE);
        
        // Apply preprocessing if available
        if (preprocessor_ != nullptr) {
            features_array = applyPreprocessing(features_array);
            if (!features_array) {
                std::cerr << "Preprocessing failed, using raw features" << std::endl;
                // Recreate features_array if preprocessing failed
                features_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, 
                                                          const_cast<double*>(features.data()));
                PyArray_ENABLEFLAGS((PyArrayObject*)features_array, NPY_ARRAY_WRITEABLE);
            }
        }
        
        // Call model.predict_proba()
        PyObject* predict_proba = PyObject_GetAttrString(model_, "predict_proba");
        if (!predict_proba || !PyCallable_Check(predict_proba)) {
            Py_DECREF(features_array);
            std::cerr << "Model does not have predict_proba method" << std::endl;
            return 0.0;
        }
        
        PyObject* args = PyTuple_New(1);
        Py_INCREF(features_array);  // Increment ref before passing
        PyTuple_SetItem(args, 0, features_array);
        
        PyObject* result = PyObject_CallObject(predict_proba, args);
        Py_DECREF(args);
        Py_DECREF(predict_proba);
        
        if (!result) {
            PyErr_Print();
            std::cerr << "Failed to call predict_proba" << std::endl;
            return 0.0;
        }
        
        // Extract probability of attack class (class 1)
        // result is a 2D array: [[prob_class0, prob_class1]]
        if (PyArray_Check(result)) {
            PyArrayObject* result_array = (PyArrayObject*)result;
            if (PyArray_NDIM(result_array) == 2 && PyArray_DIM(result_array, 1) >= 2) {
                double* data = (double*)PyArray_DATA(result_array);
                double attack_prob = data[1];  // Probability of attack class
                Py_DECREF(result);
                return attack_prob;
            }
        }
        
        Py_DECREF(result);
        return 0.0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in predict: " << e.what() << std::endl;
        return 0.0;
    }
}

std::vector<double> MLInferenceEngine::predictBatch(const std::vector<std::vector<double>>& features_batch) {
    std::vector<double> results;
    results.reserve(features_batch.size());
    
    if (!model_loaded_ || features_batch.empty()) {
        return results;
    }
    
    try {
        // Convert batch to numpy array
        size_t batch_size = features_batch.size();
        size_t feature_size = features_batch[0].size();
        
        // Flatten features into contiguous array
        std::vector<double> flat_features;
        flat_features.reserve(batch_size * feature_size);
        for (const auto& features : features_batch) {
            if (features.size() != feature_size) {
                std::cerr << "Inconsistent feature sizes in batch" << std::endl;
                return results;
            }
            flat_features.insert(flat_features.end(), features.begin(), features.end());
        }
        
        // Create 2D numpy array
        npy_intp dims[2] = {static_cast<npy_intp>(batch_size), static_cast<npy_intp>(feature_size)};
        PyObject* features_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, flat_features.data());
        if (!features_array) {
            PyErr_Print();
            std::cerr << "Failed to create numpy array for batch" << std::endl;
            return results;
        }
        
        PyArray_ENABLEFLAGS((PyArrayObject*)features_array, NPY_ARRAY_WRITEABLE);
        
        // Apply preprocessing if available
        if (preprocessor_ != nullptr) {
            features_array = applyPreprocessing(features_array);
            if (!features_array) {
                std::cerr << "Preprocessing failed, using raw features" << std::endl;
                // Recreate features_array if preprocessing failed
                features_array = PyArray_SimpleNewFromData(2, dims, NPY_FLOAT64, flat_features.data());
                PyArray_ENABLEFLAGS((PyArrayObject*)features_array, NPY_ARRAY_WRITEABLE);
            }
        }
        
        // Call model.predict_proba()
        PyObject* predict_proba = PyObject_GetAttrString(model_, "predict_proba");
        if (!predict_proba || !PyCallable_Check(predict_proba)) {
            Py_DECREF(features_array);
            std::cerr << "Model does not have predict_proba method" << std::endl;
            return results;
        }
        
        PyObject* args = PyTuple_New(1);
        Py_INCREF(features_array);  // Increment ref before passing
        PyTuple_SetItem(args, 0, features_array);
        
        PyObject* result = PyObject_CallObject(predict_proba, args);
        Py_DECREF(args);
        Py_DECREF(predict_proba);
        
        if (!result) {
            PyErr_Print();
            std::cerr << "Failed to call predict_proba on batch" << std::endl;
            return results;
        }
        
        // Extract probabilities of attack class (class 1)
        if (PyArray_Check(result)) {
            PyArrayObject* result_array = (PyArrayObject*)result;
            if (PyArray_NDIM(result_array) == 2 && PyArray_DIM(result_array, 1) >= 2) {
                double* data = (double*)PyArray_DATA(result_array);
                for (size_t i = 0; i < batch_size; ++i) {
                    results.push_back(data[i * 2 + 1]);  // Probability of attack class
                }
            }
        }
        
        Py_DECREF(result);
        return results;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception in predictBatch: " << e.what() << std::endl;
        return results;
    }
}

PyObject* MLInferenceEngine::applyPreprocessing(PyObject* features_array) {
    if (!preprocessor_ || !features_array) {
        return features_array;
    }
    
    try {
        // Call preprocessor.transform()
        PyObject* transform_method = PyObject_GetAttrString(preprocessor_, "transform");
        if (!transform_method || !PyCallable_Check(transform_method)) {
            if (transform_method) Py_DECREF(transform_method);
            return features_array;
        }
        
        PyObject* args = PyTuple_New(1);
        Py_INCREF(features_array);  // Increment ref before passing
        PyTuple_SetItem(args, 0, features_array);
        
        PyObject* result = PyObject_CallObject(transform_method, args);
        Py_DECREF(args);
        Py_DECREF(transform_method);
        
        if (result) {
            return result;
        } else {
            PyErr_Clear();
            return features_array;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception in preprocessing: " << e.what() << std::endl;
        return features_array;
    }
}
