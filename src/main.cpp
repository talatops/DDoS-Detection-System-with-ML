#include <iostream>
#include <string>
#include <unordered_map>
#include "ingest/pcap_reader.h"
#include "ingest/window_manager.h"
#include "detectors/entropy_cpu.h"
#include "detectors/decision_engine.h"
#include "opencl/gpu_detector.h"
#include "ml/inference_engine.h"
#include "ml/feature_builder.h"
#include "blocking/rtbh_controller.h"
#include "blocking/pcap_filter.h"
#include "utils/logger.h"
#include "utils/config_loader.h"
#include "utils/metrics_collector.h"
#include <chrono>
#include <vector>
#include <string>
#include <algorithm>
#include <arpa/inet.h>

int main(int argc, char* argv[]) {
    std::string pcap_file = "data/cic-ddos2019/ddostrace.20070804_145436.pcap";
    uint32_t window_size = 1;  // 1 second
    size_t batch_size = 128;
    bool use_gpu = true;  // Default to GPU
    std::string dashboard_url = "";  // Optional dashboard URL
    std::string detection_config_path = "config/detection_config.json";
    std::string model_manifest_path = "models/model_manifest.json";
    std::string requested_model = "auto";
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--pcap" && i + 1 < argc) {
            pcap_file = argv[++i];
        } else if (arg == "--window" && i + 1 < argc) {
            window_size = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        } else if (arg == "--use-gpu") {
            use_gpu = true;
        } else if (arg == "--use-cpu") {
            use_gpu = false;
        } else if (arg == "--dashboard-url" && i + 1 < argc) {
            dashboard_url = argv[++i];
        } else if (arg == "--detection-config" && i + 1 < argc) {
            detection_config_path = argv[++i];
        } else if (arg == "--model-manifest" && i + 1 < argc) {
            model_manifest_path = argv[++i];
        } else if (arg == "--ml-model" && i + 1 < argc) {
            requested_model = argv[++i];
        }
    }
    
    std::cout << "=== DDoS Detection System ===" << std::endl;
    std::cout << "PCAP: " << pcap_file << std::endl;
    std::cout << "Window: " << window_size << "s" << std::endl;
    std::cout << "Batch: " << batch_size << std::endl;
    std::cout << "Mode: " << (use_gpu ? "GPU" : "CPU") << std::endl;
    if (!dashboard_url.empty()) {
        std::cout << "Dashboard URL: " << dashboard_url << std::endl;
    }
    
    // Initialize components
    Logger logger;
    if (!logger.initialize("logs")) {
        std::cerr << "Failed to initialize logger" << std::endl;
        return 1;
    }
    
    MetricsCollector metrics_collector;
    metrics_collector.setLogger(&logger);
    metrics_collector.start("logs");
    
    // Initialize GPU detector (if requested)
    GPUDetector* gpu_detector = nullptr;
    bool gpu_available = false;
    
    if (use_gpu) {
        gpu_detector = new GPUDetector();
        if (gpu_detector->initialize()) {
            gpu_detector->setBatchSize(batch_size);
            gpu_available = true;
            std::cout << "GPU detector initialized successfully" << std::endl;
        } else {
            std::cerr << "WARNING: Failed to initialize GPU detector, falling back to CPU" << std::endl;
            delete gpu_detector;
            gpu_detector = nullptr;
            use_gpu = false;
        }
    }
    
    // Initialize CPU entropy detector (always available as fallback)
    EntropyDetector entropy_detector;
    
    // Initialize ML inference engine
    MLInferenceEngine* ml_engine = nullptr;
    bool ml_available = false;
    std::string model_path = "models/rf_model.joblib";
    std::string preprocessor_path = "models/preprocessor.joblib";
    std::vector<std::string> model_imports;
    std::string selected_model_name = "rf";
    
    MLModelRegistry registry;
    if (loadModelRegistry(model_manifest_path, registry)) {
        std::string resolved_name;
        const ModelInfo* info = resolveModel(registry, requested_model, resolved_name);
        if (info) {
            selected_model_name = resolved_name;
            if (!info->path.empty()) {
                model_path = info->path;
            }
            if (!info->preprocessor.empty()) {
                preprocessor_path = info->preprocessor;
            }
            model_imports = info->imports;
        }
    }
    
    ml_engine = new MLInferenceEngine();
    if (ml_engine->loadModel(model_path, preprocessor_path, model_imports)) {
        ml_available = true;
        std::cout << "ML model loaded successfully from: " << model_path << std::endl;
        metrics_collector.setModelName(selected_model_name);
    } else {
        std::cerr << "WARNING: Failed to load ML model from: " << model_path << std::endl;
        std::cerr << "         Continuing without ML inference" << std::endl;
        selected_model_name = "disabled";
        metrics_collector.setModelName(selected_model_name);
        delete ml_engine;
        ml_engine = nullptr;
    }
    
    // Initialize feature builder
    FeatureBuilder feature_builder;
    
    DetectionConfig detection_cfg;
    if (!loadDetectionConfig(detection_config_path, detection_cfg)) {
        std::cerr << "WARNING: Unable to load detection config from "
                  << detection_config_path << ". Using defaults." << std::endl;
    }
    
    DecisionEngine decision_engine(detection_cfg.entropy_threshold,
                                   detection_cfg.ml_threshold,
                                   detection_cfg.cusum_threshold,
                                   detection_cfg.pca_threshold,
                                   detection_cfg.use_weighted);
    decision_engine.setWeights(detection_cfg.w_entropy,
                               detection_cfg.w_ml,
                               detection_cfg.w_cusum,
                               detection_cfg.w_pca);
    
    // Initialize blocking
    RTBHController rtbh_controller;
    PcapFilter pcap_filter(&rtbh_controller);
    
    // Open pcap file
    PcapReader pcap_reader;
    if (!pcap_reader.open(pcap_file)) {
        std::cerr << "Failed to open pcap file" << std::endl;
        return 1;
    }
    
    // Create window manager
    WindowManager window_manager(window_size);
    
    std::vector<WindowStats> window_batch;
    PacketInfo packet;
    uint64_t packet_count = 0;
    uint64_t window_count = 0;
    
    auto getTopSrcIp = [](const WindowStats& stats) -> uint32_t {
        uint32_t top_ip = 0;
        uint32_t max_count = 0;
        for (const auto& entry : stats.src_ip_counts) {
            if (entry.second > max_count) {
                max_count = entry.second;
                top_ip = entry.first;
            }
        }
        return top_ip;
    };
    
    auto processCurrentBatch = [&]() {
        if (window_batch.empty()) {
            return;
        }
        std::vector<double> entropy_results;
        
        auto logResult = [&](const DetectionResult& result, double ml_prob, const WindowStats& stats, uint64_t index) {
            if (!result.is_attack) {
                return;
            }
            std::cout << "ATTACK DETECTED in window " << index
                      << " - Triggers: " << result.detector_triggered << std::endl;
            uint64_t timestamp_ms = stats.window_end_us / 1000ULL;
            uint64_t window_start_ms = stats.window_start_us / 1000ULL;
            uint32_t top_src_ip = getTopSrcIp(stats);
            logger.logAlert(timestamp_ms, window_start_ms, index, top_src_ip,
                            result.entropy_score, ml_prob,
                            result.cusum_score, result.pca_score,
                            result.combined_score, result.detector_triggered,
                            selected_model_name);
            
            // Add IP to blackhole list and log blocking event
            if (top_src_ip != 0) {
                struct in_addr addr;
                addr.s_addr = top_src_ip;
                std::string ip_str = std::string(inet_ntoa(addr));
                
                // Check if already blackholed to avoid duplicate logging
                bool was_blackholed = rtbh_controller.isBlackholed(top_src_ip);
                if (rtbh_controller.addIP(top_src_ip)) {
                    uint64_t impacted_packets = stats.total_packets;
                    // Don't count these packets as dropped since they were already processed
                    // Dropped packets will be logged in subsequent windows when filtering occurs
                    uint64_t dropped_packets = 0;
                    logger.logBlocking(timestamp_ms, top_src_ip, impacted_packets, dropped_packets);
                    std::cout << "  -> IP " << ip_str << " added to blackhole list (will drop future packets)" << std::endl;
                }
            }
        };
        
        if (gpu_available && gpu_detector) {
            if (gpu_detector->processBatch(window_batch, entropy_results)) {
                auto now_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                    std::chrono::system_clock::now().time_since_epoch()).count();
                double kernel_ms = gpu_detector->getKernelTime();
                logger.logKernelTime(static_cast<uint64_t>(now_ms),
                                     "compute_multi_entropy",
                                     kernel_ms);
                size_t num_features = 6;
                std::vector<std::vector<double>> ml_feature_vectors;
                std::vector<double> ml_probabilities;
                
                for (size_t i = 0; i < window_batch.size(); ++i) {
                    size_t base_idx = i * num_features;
                    if (base_idx + 5 < entropy_results.size()) {
                        std::vector<double> window_gpu_entropy(
                            entropy_results.begin() + base_idx,
                            entropy_results.begin() + base_idx + 6
                        );
                        std::vector<double> ml_features;
                        if (feature_builder.buildFeatures(window_batch[i], window_gpu_entropy, ml_features)) {
                            ml_feature_vectors.push_back(ml_features);
                        }
                    }
                }
                
                if (ml_available && ml_engine && !ml_feature_vectors.empty()) {
                    ml_probabilities = ml_engine->predictBatch(ml_feature_vectors);
                } else {
                    ml_probabilities.assign(window_batch.size(), 0.5);
                }
                
                size_t ml_idx = 0;
                for (size_t i = 0; i < window_batch.size(); ++i) {
                    size_t base_idx = i * num_features;
                    if (base_idx + 5 < entropy_results.size()) {
                        EntropyDetector::EntropyFeatures features;
                        features.src_ip_entropy = entropy_results[base_idx + 0];
                        features.dst_ip_entropy = entropy_results[base_idx + 1];
                        features.src_port_entropy = entropy_results[base_idx + 2];
                        features.dst_port_entropy = entropy_results[base_idx + 3];
                        features.packet_size_entropy = entropy_results[base_idx + 4];
                        features.protocol_entropy = entropy_results[base_idx + 5];
                        
                        double ml_prob = (ml_idx < ml_probabilities.size()) ? ml_probabilities[ml_idx++] : 0.5;
                        DetectionResult result = decision_engine.detect(features, ml_prob);
                        uint64_t window_index = (window_count - window_batch.size() + i);
                        logResult(result, ml_prob, window_batch[i], window_index);
                    }
                }
            } else {
                std::cerr << "WARNING: GPU batch processing failed, falling back to CPU" << std::endl;
                std::vector<std::vector<double>> ml_feature_vectors;
                for (const auto& win : window_batch) {
                    EntropyDetector::EntropyFeatures features = entropy_detector.calculateFeatures(win);
                    std::vector<double> ml_features;
                    if (feature_builder.buildFeaturesFromCPU(win, features, ml_features)) {
                        ml_feature_vectors.push_back(ml_features);
                    }
                }
                std::vector<double> ml_probabilities;
                if (ml_available && ml_engine && !ml_feature_vectors.empty()) {
                    ml_probabilities = ml_engine->predictBatch(ml_feature_vectors);
                } else {
                    ml_probabilities.assign(window_batch.size(), 0.5);
                }
                size_t ml_idx = 0;
                for (size_t i = 0; i < window_batch.size(); ++i) {
                    EntropyDetector::EntropyFeatures features = entropy_detector.calculateFeatures(window_batch[i]);
                    double ml_prob = (ml_idx < ml_probabilities.size()) ? ml_probabilities[ml_idx++] : 0.5;
                    DetectionResult result = decision_engine.detect(features, ml_prob);
                    uint64_t window_index = (window_count - window_batch.size() + i);
                    logResult(result, ml_prob, window_batch[i], window_index);
                }
            }
        } else {
            std::vector<std::vector<double>> ml_feature_vectors;
            for (const auto& win : window_batch) {
                EntropyDetector::EntropyFeatures features = entropy_detector.calculateFeatures(win);
                std::vector<double> ml_features;
                if (feature_builder.buildFeaturesFromCPU(win, features, ml_features)) {
                    ml_feature_vectors.push_back(ml_features);
                }
            }
            std::vector<double> ml_probabilities;
            if (ml_available && ml_engine && !ml_feature_vectors.empty()) {
                ml_probabilities = ml_engine->predictBatch(ml_feature_vectors);
            } else {
                ml_probabilities.assign(window_batch.size(), 0.5);
            }
            size_t ml_idx = 0;
            for (size_t i = 0; i < window_batch.size(); ++i) {
                EntropyDetector::EntropyFeatures features = entropy_detector.calculateFeatures(window_batch[i]);
                double ml_prob = (ml_idx < ml_probabilities.size()) ? ml_probabilities[ml_idx++] : 0.5;
                DetectionResult result = decision_engine.detect(features, ml_prob);
                uint64_t window_index = (window_count - window_batch.size() + i);
                logResult(result, ml_prob, window_batch[i], window_index);
            }
        }
        
        window_batch.clear();
        metrics_collector.setWindowsProcessed(window_count);
    };
    
    // Window callback to collect windows for batch processing
    window_manager.setWindowCallback([&](const WindowStats& stats) {
        window_batch.push_back(stats);
        window_count++;
        
        if (window_batch.size() >= batch_size) {
            processCurrentBatch();
        }
    });
    
    // Process packets
    std::cout << "Processing packets..." << std::endl;
    uint64_t dropped_packets_total = 0;
    std::unordered_map<uint32_t, uint64_t> dropped_packets_per_ip; // Track dropped packets per IP
    uint64_t last_blocking_log_time = 0;
    const uint64_t BLOCKING_LOG_INTERVAL_MS = 1000; // Log blocking stats every second
    
    while (pcap_reader.readNextPacket(packet)) {
        if (!packet.is_valid) continue;
        
        // Check if packet should be dropped (filter blackholed IPs)
        if (pcap_filter.shouldDrop(packet.src_ip, packet.dst_ip)) {
            dropped_packets_total++;
            // Track which IP caused the drop (prioritize source IP as that's what we block)
            if (rtbh_controller.isBlackholed(packet.src_ip)) {
                dropped_packets_per_ip[packet.src_ip]++;
            } else if (rtbh_controller.isBlackholed(packet.dst_ip)) {
                dropped_packets_per_ip[packet.dst_ip]++;
            }
            continue; // Skip this packet, don't add to window
        }
        
        packet_count++;
        
        // Add to window
        window_manager.addPacket(packet);
        
        // Check if window should be closed (triggers callback)
        window_manager.checkWindow(packet.timestamp_us);
        
        // Periodically log blocking statistics
        uint64_t current_time_ms = packet.timestamp_us / 1000ULL;
        if (current_time_ms - last_blocking_log_time >= BLOCKING_LOG_INTERVAL_MS && !dropped_packets_per_ip.empty()) {
            // Log blocking stats for all IPs that had dropped packets
            for (const auto& entry : dropped_packets_per_ip) {
                if (entry.second > 0) {
                    logger.logBlocking(current_time_ms, entry.first, 0, entry.second);
                }
            }
            dropped_packets_per_ip.clear(); // Reset for next period
            last_blocking_log_time = current_time_ms;
        }
        
        if (packet_count % 10000 == 0) {
            std::cout << "Processed " << packet_count << " packets, " 
                     << window_count << " windows" << std::endl;
        }
    }
    
    // Close final window if there are remaining packets
    window_manager.closeWindow();  // Force close final window
    
    // Process remaining windows in batch
    if (!window_batch.empty()) {
        processCurrentBatch();
    }
    
    // Log any remaining dropped packets
    if (!dropped_packets_per_ip.empty()) {
        uint64_t final_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
        for (const auto& entry : dropped_packets_per_ip) {
            if (entry.second > 0) {
                logger.logBlocking(final_time_ms, entry.first, 0, entry.second);
            }
        }
    }
    
    std::cout << "Processed " << packet_count << " packets total" << std::endl;
    std::cout << "Dropped " << dropped_packets_total << " packets (blackholed IPs)" << std::endl;
    std::cout << "Processed " << window_count << " windows total" << std::endl;
    
    if (gpu_available && gpu_detector) {
        std::cout << "GPU kernel execution time: " << gpu_detector->getKernelTime() << " ms" << std::endl;
    }
    
    // Cleanup
    if (gpu_detector) {
        delete gpu_detector;
    }
    if (ml_engine) {
        delete ml_engine;
    }
    metrics_collector.stop();
    logger.close();
    
    return 0;
}

