#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "../src/detectors/entropy_cpu.h"
#include "../src/ingest/window_manager.h"
#include "../src/opencl/gpu_detector.h"
#include "../src/utils/simple_json.h"

namespace {

template <typename MapType, typename KeyType>
void loadCounts(const SimpleJson::Value* value, MapType& dest) {
    if (!value || !value->isArray()) {
        return;
    }
    for (const auto& entry : value->array_values) {
        if (!entry.isObject()) {
            continue;
        }
        const auto* key_val = entry.find("key");
        const auto* count_val = entry.find("value");
        if (!key_val || !count_val || !key_val->isNumber() || !count_val->isNumber()) {
            continue;
        }
        KeyType key = static_cast<KeyType>(key_val->number_value);
        uint32_t count = static_cast<uint32_t>(count_val->number_value);
        dest[key] = count;
    }
}

bool loadWindowsFromJson(const std::string& path, std::vector<WindowStats>& windows) {
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Unable to open window file: " << path << std::endl;
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string error;
    SimpleJson::Value root;
    if (!SimpleJson::parse(buffer.str(), root, error)) {
        std::cerr << "ERROR: Failed to parse JSON: " << error << std::endl;
        return false;
    }
    
    const SimpleJson::Value* windows_val = root.find("windows");
    if (!windows_val || !windows_val->isArray()) {
        std::cerr << "ERROR: JSON must contain 'windows' array" << std::endl;
        return false;
    }
    
    for (const auto& window_val : windows_val->array_values) {
        if (!window_val.isObject()) {
            continue;
        }
        WindowStats window;
        
        if (const auto* start = window_val.find("window_start_us"); start && start->isNumber()) {
            window.window_start_us = static_cast<uint64_t>(start->number_value);
        }
        if (const auto* end = window_val.find("window_end_us"); end && end->isNumber()) {
            window.window_end_us = static_cast<uint64_t>(end->number_value);
        }
        if (const auto* packets = window_val.find("total_packets"); packets && packets->isNumber()) {
            window.total_packets = static_cast<uint32_t>(packets->number_value);
        }
        if (const auto* bytes = window_val.find("total_bytes"); bytes && bytes->isNumber()) {
            window.total_bytes = static_cast<uint64_t>(bytes->number_value);
        }
        
        loadCounts<std::unordered_map<uint32_t, uint32_t>, uint32_t>(
            window_val.find("src_ip_counts"), window.src_ip_counts);
        loadCounts<std::unordered_map<uint32_t, uint32_t>, uint32_t>(
            window_val.find("dst_ip_counts"), window.dst_ip_counts);
        loadCounts<std::unordered_map<uint16_t, uint32_t>, uint16_t>(
            window_val.find("src_port_counts"), window.src_port_counts);
        loadCounts<std::unordered_map<uint16_t, uint32_t>, uint16_t>(
            window_val.find("dst_port_counts"), window.dst_port_counts);
        loadCounts<std::unordered_map<uint16_t, uint32_t>, uint16_t>(
            window_val.find("packet_size_counts"), window.packet_size_counts);
        loadCounts<std::unordered_map<uint8_t, uint32_t>, uint8_t>(
            window_val.find("protocol_counts"), window.protocol_counts);
        
        windows.push_back(std::move(window));
    }
    
    if (windows.empty()) {
        std::cerr << "ERROR: No windows parsed from JSON" << std::endl;
        return false;
    }
    return true;
}

std::string buildResultsJson(const std::vector<std::vector<double>>& cpu_results,
                             const std::vector<double>& gpu_results,
                             double max_diff,
                             bool all_match) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(10);
    oss << "{";
    oss << "\"status\":\"" << (all_match ? "ok" : "mismatch") << "\",";
    oss << "\"max_diff\":" << max_diff << ",";
    oss << "\"num_windows\":" << cpu_results.size() << ",";
    oss << "\"gpu_results\":[";
    for (size_t i = 0; i < gpu_results.size(); ++i) {
        if (i > 0) {
            oss << ",";
        }
        oss << gpu_results[i];
    }
    oss << "],";
    oss << "\"cpu_results\":[";
    const size_t num_features = 6;
    for (size_t i = 0; i < cpu_results.size(); ++i) {
        for (size_t j = 0; j < num_features; ++j) {
            if (i != 0 || j != 0) {
                oss << ",";
            }
            oss << cpu_results[i][j];
        }
    }
    oss << "]";
    oss << "}";
    return oss.str();
}

void writeOutput(const std::string& data, const std::string& output_path) {
    if (output_path.empty()) {
        std::cout << data << std::endl;
        return;
    }
    std::ofstream file(output_path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Unable to open output file: " << output_path << std::endl;
        return;
    }
    file << data;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string output_path;
    if (argc >= 2) {
        input_path = argv[1];
    }
    if (argc >= 3) {
        output_path = argv[2];
    }
    
    std::vector<WindowStats> windows;
    if (!input_path.empty()) {
        if (!loadWindowsFromJson(input_path, windows)) {
            return 1;
        }
    } else {
        std::cerr << "WARNING: No input file provided, generating synthetic windows." << std::endl;
        for (int i = 0; i < 10; ++i) {
            WindowStats window;
            window.total_packets = 1000 + i * 100;
            window.total_bytes = window.total_packets * 1000;
            window.src_ip_counts[0x01010101] = static_cast<uint32_t>(window.total_packets * 0.9);
            window.src_ip_counts[0x02020202] = static_cast<uint32_t>(window.total_packets * 0.1);
            window.dst_ip_counts[0xC0A80101] = window.total_packets;
            window.src_port_counts[80] = static_cast<uint32_t>(window.total_packets * 0.8);
            window.src_port_counts[443] = static_cast<uint32_t>(window.total_packets * 0.2);
            window.dst_port_counts[80] = window.total_packets;
            window.packet_size_counts[64] = window.total_packets / 2;
            window.packet_size_counts[1500] = window.total_packets / 2;
            window.protocol_counts[6] = window.total_packets;
            windows.push_back(window);
        }
    }
    
    GPUDetector gpu_detector;
    if (!gpu_detector.initialize()) {
        std::cerr << "Failed to initialize GPU detector" << std::endl;
        return 1;
    }
    
    EntropyDetector cpu_detector;
    std::vector<std::vector<double>> cpu_results;
    cpu_results.reserve(windows.size());
    for (const auto& window : windows) {
        EntropyDetector::EntropyFeatures features = cpu_detector.calculateFeatures(window);
        cpu_results.push_back({
            features.src_ip_entropy,
            features.dst_ip_entropy,
            features.src_port_entropy,
            features.dst_port_entropy,
            features.packet_size_entropy,
            features.protocol_entropy
        });
    }
    
    std::vector<double> gpu_results;
    if (!gpu_detector.processBatch(windows, gpu_results)) {
        std::cerr << "GPU batch processing failed" << std::endl;
        return 1;
    }
    
    const size_t num_features = 6;
    bool all_match = true;
    double max_diff = 0.0;
    for (size_t i = 0; i < windows.size(); ++i) {
        size_t base_idx = i * num_features;
        if (base_idx + num_features > gpu_results.size()) {
            std::cerr << "ERROR: GPU results incomplete for window " << i << std::endl;
            return 1;
        }
        for (size_t j = 0; j < num_features; ++j) {
            double cpu_val = cpu_results[i][j];
            double gpu_val = gpu_results[base_idx + j];
            double diff = std::abs(cpu_val - gpu_val);
            if (diff > max_diff) {
                max_diff = diff;
            }
            if (diff > 0.01) {
                all_match = false;
            }
        }
    }
    
    std::string json_output = buildResultsJson(cpu_results, gpu_results, max_diff, all_match);
    writeOutput(json_output, output_path);
    return all_match ? 0 : 1;
}

