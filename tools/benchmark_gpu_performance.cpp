#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <iomanip>
#include "../src/opencl/gpu_detector.h"
#include "../src/detectors/entropy_cpu.h"
#include "../src/ingest/window_manager.h"

double calculate_cpu_entropy_time(const std::vector<WindowStats>& windows) {
    EntropyDetector detector;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (const auto& window : windows) {
        EntropyDetector::EntropyFeatures features = detector.calculateFeatures(window);
        (void)features;  // Suppress unused variable warning
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;  // Convert to milliseconds
}

double calculate_gpu_entropy_time(const std::vector<WindowStats>& windows, GPUDetector& gpu_detector) {
    std::vector<double> entropy_results;
    auto start = std::chrono::high_resolution_clock::now();
    
    if (!gpu_detector.processBatch(windows, entropy_results)) {
        return -1.0;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0;  // Convert to milliseconds
}

WindowStats generate_test_window(size_t num_packets, bool ddos_like = false) {
    WindowStats window;
    window.total_packets = num_packets;
    window.total_bytes = num_packets * 1000;  // Average 1000 bytes per packet
    
    std::random_device rd;
    std::mt19937 gen(rd());
    
    if (ddos_like) {
        // DDoS-like: few source IPs, many packets
        window.src_ip_counts[1] = num_packets * 0.9;
        window.src_ip_counts[2] = num_packets * 0.1;
        window.dst_ip_counts[100] = num_packets;
        window.src_port_counts[80] = num_packets * 0.8;
        window.src_port_counts[443] = num_packets * 0.2;
        window.dst_port_counts[80] = num_packets;
    } else {
        // Normal traffic: many unique IPs/ports
        std::uniform_int_distribution<uint32_t> ip_dist(1, 1000);
        std::uniform_int_distribution<uint16_t> port_dist(1024, 65535);
        
        for (size_t i = 0; i < num_packets; ++i) {
            window.src_ip_counts[ip_dist(gen)]++;
            window.dst_ip_counts[ip_dist(gen)]++;
            window.src_port_counts[port_dist(gen)]++;
            window.dst_port_counts[port_dist(gen)]++;
        }
    }
    
    // Packet sizes and protocols
    window.packet_size_counts[64] = num_packets / 3;
    window.packet_size_counts[1500] = num_packets / 3;
    window.packet_size_counts[576] = num_packets / 3;
    window.protocol_counts[6] = num_packets / 2;   // TCP
    window.protocol_counts[17] = num_packets / 2; // UDP
    
    return window;
}

std::vector<WindowStats> generate_test_batch(size_t batch_size, size_t packets_per_window, bool ddos_like = false) {
    std::vector<WindowStats> windows;
    windows.reserve(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        windows.push_back(generate_test_window(packets_per_window, ddos_like));
    }
    
    return windows;
}

void benchmark_batch_size(size_t batch_size, GPUDetector& gpu_detector) {
    const size_t packets_per_window = 1000;
    
    std::cout << "\n  Batch size: " << batch_size << " windows" << std::endl;
    
    // Generate test data
    auto windows = generate_test_batch(batch_size, packets_per_window, false);
    
    // CPU benchmark
    double cpu_time = calculate_cpu_entropy_time(windows);
    std::cout << "    CPU time: " << std::fixed << std::setprecision(3) 
              << cpu_time << " ms" << std::endl;
    
    // GPU benchmark
    double gpu_time = calculate_gpu_entropy_time(windows, gpu_detector);
    if (gpu_time < 0) {
        std::cout << "    GPU time: FAILED" << std::endl;
        return;
    }
    
    double kernel_time = gpu_detector.getKernelTime();
    std::cout << "    GPU time (total): " << std::fixed << std::setprecision(3) 
              << gpu_time << " ms" << std::endl;
    std::cout << "    GPU kernel time: " << std::fixed << std::setprecision(3) 
              << kernel_time << " ms" << std::endl;
    
    // Calculate speedup
    if (cpu_time > 0 && gpu_time > 0) {
        double speedup = cpu_time / gpu_time;
        std::cout << "    Speedup: " << std::fixed << std::setprecision(2) 
                  << speedup << "x" << std::endl;
    }
}

int main() {
    std::cout << "=" << std::string(78, '=') << std::endl;
    std::cout << "GPU Performance Benchmark" << std::endl;
    std::cout << "=" << std::string(78, '=') << std::endl;
    
    // Initialize GPU detector
    std::cout << "\nInitializing GPU detector..." << std::endl;
    GPUDetector gpu_detector;
    if (!gpu_detector.initialize()) {
        std::cerr << "ERROR: Failed to initialize GPU detector" << std::endl;
        std::cerr << "Make sure NVIDIA GPU and OpenCL drivers are installed" << std::endl;
        return 1;
    }
    std::cout << "GPU detector initialized successfully" << std::endl;
    
    // Benchmark different batch sizes
    std::vector<size_t> batch_sizes = {32, 64, 128, 256, 512};
    
    std::cout << "\n" << std::string(78, '-') << std::endl;
    std::cout << "Benchmarking different batch sizes:" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    
    for (size_t batch_size : batch_sizes) {
        benchmark_batch_size(batch_size, gpu_detector);
    }
    
    // Benchmark DDoS-like traffic
    std::cout << "\n" << std::string(78, '-') << std::endl;
    std::cout << "Benchmarking DDoS-like traffic (batch size 256):" << std::endl;
    std::cout << std::string(78, '-') << std::endl;
    
    auto ddos_windows = generate_test_batch(256, 1000, true);
    double cpu_time_ddos = calculate_cpu_entropy_time(ddos_windows);
    double gpu_time_ddos = calculate_gpu_entropy_time(ddos_windows, gpu_detector);
    
    std::cout << "  CPU time: " << std::fixed << std::setprecision(3) 
              << cpu_time_ddos << " ms" << std::endl;
    std::cout << "  GPU time: " << std::fixed << std::setprecision(3) 
              << gpu_time_ddos << " ms" << std::endl;
    if (cpu_time_ddos > 0 && gpu_time_ddos > 0) {
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2) 
                  << (cpu_time_ddos / gpu_time_ddos) << "x" << std::endl;
    }
    
    std::cout << "\n" << std::string(78, '=') << std::endl;
    std::cout << "Benchmark complete" << std::endl;
    std::cout << std::string(78, '=') << std::endl;
    
    return 0;
}

