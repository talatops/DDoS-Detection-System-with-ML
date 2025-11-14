#include <iostream>
#include <string>
#include "ingest/pcap_reader.h"
#include "ingest/window_manager.h"
#include "detectors/entropy_cpu.h"
#include "detectors/decision_engine.h"
#include "opencl/gpu_detector.h"
#include "blocking/rtbh_controller.h"
#include "blocking/pcap_filter.h"
#include "utils/logger.h"
#include "utils/metrics_collector.h"
#include <chrono>
#include <vector>

int main(int argc, char* argv[]) {
    std::string pcap_file = "data/cic-ddos2019/ddostrace.20070804_145436.pcap";
    uint32_t window_size = 1;  // 1 second
    size_t batch_size = 128;
    
    // Parse command line arguments (simplified)
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--pcap" && i + 1 < argc) {
            pcap_file = argv[++i];
        } else if (arg == "--window" && i + 1 < argc) {
            window_size = std::stoi(argv[++i]);
        } else if (arg == "--batch" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
        }
    }
    
    std::cout << "=== DDoS Detection System ===" << std::endl;
    std::cout << "PCAP: " << pcap_file << std::endl;
    std::cout << "Window: " << window_size << "s" << std::endl;
    std::cout << "Batch: " << batch_size << std::endl;
    
    // Initialize components
    Logger logger;
    if (!logger.initialize("logs")) {
        std::cerr << "Failed to initialize logger" << std::endl;
        return 1;
    }
    
    MetricsCollector metrics_collector;
    metrics_collector.start("logs");
    
    // Initialize GPU detector
    GPUDetector gpu_detector;
    if (!gpu_detector.initialize()) {
        std::cerr << "Failed to initialize GPU detector" << std::endl;
        return 1;
    }
    
    // Initialize detectors
    EntropyDetector entropy_detector;
    DecisionEngine decision_engine;
    
    // Initialize blocking
    RTBHController rtbh_controller;
    PcapFilter pcap_filter(&rtbh_controller);
    
    // Open pcap file
    PcapReader pcap_reader;
    if (!pcap_reader.open(pcap_file)) {
        std::cerr << "Failed to open pcap file" << std::endl;
        return 1;
    }
    
    // Create window managers (1s and 5s)
    WindowManager window_1s(1);
    WindowManager window_5s(5);
    
    std::vector<WindowStats> window_batch;
    PacketInfo packet;
    uint64_t packet_count = 0;
    
    // Process packets
    std::cout << "Processing packets..." << std::endl;
    while (pcap_reader.readNextPacket(packet)) {
        if (!packet.is_valid) continue;
        
        packet_count++;
        
        // Add to windows
        window_1s.addPacket(packet);
        window_5s.addPacket(packet);
        
        // Check if windows should be closed
        window_1s.checkWindow(packet.timestamp_us);
        window_5s.checkWindow(packet.timestamp_us);
        
        // Process batch when ready
        // (Simplified - would need proper window callback)
        
        if (packet_count % 10000 == 0) {
            std::cout << "Processed " << packet_count << " packets" << std::endl;
        }
    }
    
    // Close remaining windows
    window_1s.closeWindow();
    window_5s.closeWindow();
    
    std::cout << "Processed " << packet_count << " packets total" << std::endl;
    
    // Cleanup
    metrics_collector.stop();
    logger.close();
    
    return 0;
}

