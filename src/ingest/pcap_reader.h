#ifndef PCAP_READER_H
#define PCAP_READER_H

#include <cstdint>
#include <string>
#include <functional>

struct PacketInfo {
    uint64_t timestamp_us;    // Microsecond timestamp
    uint32_t src_ip;          // Source IP address (network byte order)
    uint32_t dst_ip;          // Destination IP address (network byte order)
    uint16_t src_port;        // Source port (network byte order)
    uint16_t dst_port;        // Destination port (network byte order)
    uint8_t protocol;         // IP protocol (6=TCP, 17=UDP, etc.)
    uint16_t packet_size;     // Packet size in bytes
    uint8_t tcp_flags;        // TCP flags (SYN, FIN, etc.)
    bool is_valid;            // Whether packet was successfully parsed
};

class PcapReader {
public:
    PcapReader();
    ~PcapReader();
    
    // Open a pcap file for reading
    bool open(const std::string& filename);
    
    // Close the pcap file
    void close();
    
    // Read next packet
    bool readNextPacket(PacketInfo& packet);
    
    // Set packet filter (BPF format)
    bool setFilter(const std::string& filter);
    
    // Get total packet count (if available)
    uint64_t getPacketCount() const { return packet_count_; }
    
    // Check if file is open
    bool isOpen() const { return handle_ != nullptr; }

private:
    void* handle_;  // pcap_t* handle
    uint64_t packet_count_;
    int link_type_;  // PCAP link type (DLT_RAW, DLT_EN10MB, etc.)
    
    // Parse Ethernet/IP/TCP/UDP headers
    bool parsePacket(const uint8_t* data, uint32_t len, PacketInfo& packet);
};

#endif // PCAP_READER_H

