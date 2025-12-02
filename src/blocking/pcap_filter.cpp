#include "pcap_filter.h"

PcapFilter::PcapFilter(RTBHController* rtbh_controller)
    : rtbh_controller_(rtbh_controller),
      packets_dropped_(0), bytes_dropped_(0), packets_processed_(0) {
}

PcapFilter::~PcapFilter() {
}

bool PcapFilter::shouldDrop(uint32_t src_ip, uint32_t dst_ip) {
    packets_processed_++;
    
    // Drop if source or destination IP is blackholed
    if (rtbh_controller_ && 
        (rtbh_controller_->isBlackholed(src_ip) || rtbh_controller_->isBlackholed(dst_ip))) {
        packets_dropped_++;
        return true;
    }
    
    return false;
}

void PcapFilter::resetStats() {
    packets_dropped_ = 0;
    bytes_dropped_ = 0;
    packets_processed_ = 0;
}

void PcapFilter::updateBlackholeList() {
    // Blackhole list is updated automatically through RTBHController
    // This method can be used to force a reload if needed
}

