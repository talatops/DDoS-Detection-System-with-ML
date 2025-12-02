#include "window_manager.h"
#include <algorithm>
#include <netinet/tcp.h>
#include <netinet/in.h>

WindowManager::WindowManager(uint32_t window_size_sec) 
    : window_size_us_(window_size_sec * 1000000ULL),
      window_start_time_us_(0),
      callback_(nullptr) {
}

WindowManager::~WindowManager() {
}

uint64_t WindowManager::createFlowHash(uint32_t src_ip, uint32_t dst_ip,
                                       uint16_t src_port, uint16_t dst_port,
                                       uint8_t protocol) {
    // Create a hash from 5-tuple
    uint64_t hash = 0;
    hash ^= static_cast<uint64_t>(src_ip) << 32;
    hash ^= static_cast<uint64_t>(dst_ip);
    hash ^= static_cast<uint64_t>(src_port) << 16;
    hash ^= static_cast<uint64_t>(dst_port);
    hash ^= static_cast<uint64_t>(protocol) << 24;
    return hash;
}

void WindowManager::addPacket(const PacketInfo& packet) {
    if (!packet.is_valid) return;
    
    // Initialize window start time if this is the first packet
    if (window_start_time_us_ == 0) {
        window_start_time_us_ = packet.timestamp_us;
        current_window_.window_start_us = packet.timestamp_us;
    }
    
    // Update window end time
    current_window_.window_end_us = packet.timestamp_us;
    
    // Update histograms
    current_window_.src_ip_counts[packet.src_ip]++;
    current_window_.dst_ip_counts[packet.dst_ip]++;
    current_window_.src_port_counts[packet.src_port]++;
    current_window_.dst_port_counts[packet.dst_port]++;
    current_window_.packet_size_counts[packet.packet_size]++;
    current_window_.protocol_counts[packet.protocol]++;
    
    // Update aggregated stats
    current_window_.total_packets++;
    current_window_.total_bytes += packet.packet_size;
    
    if (packet.protocol == IPPROTO_TCP) {
        current_window_.tcp_packets++;
        uint8_t flags = packet.tcp_flags;
        if (flags & TH_SYN) current_window_.syn_packets++;
        if (flags & TH_FIN) current_window_.fin_packets++;
        if (flags & TH_RST) current_window_.rst_packets++;
        if (flags & TH_ACK) current_window_.ack_packets++;
    } else if (packet.protocol == IPPROTO_UDP) {
        current_window_.udp_packets++;
    }

    // Update unique IP counts
    current_window_.unique_src_ips = current_window_.src_ip_counts.size();
    current_window_.unique_dst_ips = current_window_.dst_ip_counts.size();
    
    // Update flow count
    uint64_t flow_hash = createFlowHash(packet.src_ip, packet.dst_ip,
                                        packet.src_port, packet.dst_port,
                                        packet.protocol);
    if (current_window_.flow_counts.find(flow_hash) == current_window_.flow_counts.end()) {
        current_window_.flow_count++;
    }
    current_window_.flow_counts[flow_hash]++;
}

void WindowManager::checkWindow(uint64_t current_time_us) {
    if (window_start_time_us_ == 0) return;
    
    uint64_t elapsed = current_time_us - window_start_time_us_;
    
    if (elapsed >= window_size_us_) {
        closeWindow();
    }
}

void WindowManager::closeWindow() {
    if (current_window_.total_packets > 0) {
        // Set final window times
        current_window_.window_start_us = window_start_time_us_;
        current_window_.window_end_us = window_start_time_us_ + window_size_us_;
        
        // Call callback if set
        if (callback_) {
            callback_(current_window_);
        }
    }
    
    // Reset for next window
    current_window_.reset();
    window_start_time_us_ = 0;
}

void WindowManager::reset() {
    closeWindow();
    current_window_.reset();
    window_start_time_us_ = 0;
}

