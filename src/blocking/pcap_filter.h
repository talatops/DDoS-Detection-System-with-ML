#ifndef PCAP_FILTER_H
#define PCAP_FILTER_H

#include <string>
#include <cstdint>
#include "blocking/rtbh_controller.h"

class PcapFilter {
public:
    PcapFilter(RTBHController* rtbh_controller);
    ~PcapFilter();
    
    // Check if packet should be dropped
    bool shouldDrop(uint32_t src_ip, uint32_t dst_ip) const;
    
    // Get statistics
    uint64_t getPacketsDropped() const { return packets_dropped_; }
    uint64_t getBytesDropped() const { return bytes_dropped_; }
    uint64_t getPacketsProcessed() const { return packets_processed_; }
    
    // Reset statistics
    void resetStats();
    
    // Update blackhole list (reload from controller)
    void updateBlackholeList();

private:
    RTBHController* rtbh_controller_;
    uint64_t packets_dropped_;
    uint64_t bytes_dropped_;
    uint64_t packets_processed_;
};

#endif // PCAP_FILTER_H

