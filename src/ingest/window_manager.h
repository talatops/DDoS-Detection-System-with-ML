#ifndef WINDOW_MANAGER_H
#define WINDOW_MANAGER_H

#include <cstdint>
#include <unordered_map>
#include <vector>
#include <functional>
#include "pcap_reader.h"

struct WindowStats {
    uint64_t window_start_us;
    uint64_t window_end_us;
    
    // Histograms for entropy calculation
    std::unordered_map<uint32_t, uint32_t> src_ip_counts;
    std::unordered_map<uint32_t, uint32_t> dst_ip_counts;
    std::unordered_map<uint16_t, uint32_t> src_port_counts;
    std::unordered_map<uint16_t, uint32_t> dst_port_counts;
    std::unordered_map<uint16_t, uint32_t> packet_size_counts;
    std::unordered_map<uint8_t, uint32_t> protocol_counts;
    
    // Aggregated statistics
    uint32_t total_packets;
    uint64_t total_bytes;
    uint32_t unique_src_ips;
    uint32_t unique_dst_ips;
    uint32_t flow_count;
    uint32_t tcp_packets;
    uint32_t udp_packets;
    uint32_t syn_packets;
    uint32_t fin_packets;
    uint32_t rst_packets;
    uint32_t ack_packets;
    
    // Flow tracking (5-tuple)
    std::unordered_map<uint64_t, uint32_t> flow_counts;  // flow_hash -> count
    
    WindowStats() : total_packets(0), total_bytes(0),
                    unique_src_ips(0), unique_dst_ips(0), flow_count(0),
                    tcp_packets(0), udp_packets(0),
                    syn_packets(0), fin_packets(0),
                    rst_packets(0), ack_packets(0) {}
    
    void reset() {
        src_ip_counts.clear();
        dst_ip_counts.clear();
        src_port_counts.clear();
        dst_port_counts.clear();
        packet_size_counts.clear();
        protocol_counts.clear();
        flow_counts.clear();
        total_packets = 0;
        total_bytes = 0;
        unique_src_ips = 0;
        unique_dst_ips = 0;
        flow_count = 0;
        tcp_packets = 0;
        udp_packets = 0;
        syn_packets = 0;
        fin_packets = 0;
        rst_packets = 0;
        ack_packets = 0;
    }
};

class WindowManager {
public:
    using WindowCallback = std::function<void(const WindowStats&)>;
    
    WindowManager(uint32_t window_size_sec = 1);
    ~WindowManager();
    
    // Add a packet to the current window
    void addPacket(const PacketInfo& packet);
    
    // Check if current window should be closed and start new one
    void checkWindow(uint64_t current_time_us);
    
    // Force close current window
    void closeWindow();
    
    // Set callback for when window is closed
    void setWindowCallback(WindowCallback callback) { callback_ = callback; }
    
    // Get current window stats
    const WindowStats& getCurrentWindow() const { return current_window_; }
    
    // Reset window manager
    void reset();

private:
    uint32_t window_size_us_;  // Window size in microseconds
    WindowStats current_window_;
    uint64_t window_start_time_us_;
    WindowCallback callback_;
    
    // Helper to create flow hash from 5-tuple
    uint64_t createFlowHash(uint32_t src_ip, uint32_t dst_ip, 
                            uint16_t src_port, uint16_t dst_port, 
                            uint8_t protocol);
};

#endif // WINDOW_MANAGER_H

