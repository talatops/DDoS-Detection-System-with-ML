#include "pcap_reader.h"
#include <pcap/pcap.h>
#include <netinet/ip.h>
#include <netinet/tcp.h>
#include <netinet/udp.h>
#include <netinet/ether.h>
#include <arpa/inet.h>
#include <cstring>
#include <iostream>

PcapReader::PcapReader() : handle_(nullptr), packet_count_(0) {
}

PcapReader::~PcapReader() {
    close();
}

bool PcapReader::open(const std::string& filename) {
    char errbuf[PCAP_ERRBUF_SIZE];
    handle_ = pcap_open_offline(filename.c_str(), errbuf);
    
    if (handle_ == nullptr) {
        std::cerr << "Error opening pcap file " << filename << ": " << errbuf << std::endl;
        return false;
    }
    
    packet_count_ = 0;
    
    // Set filter to only IP traffic
    setFilter("ip");
    
    return true;
}

void PcapReader::close() {
    if (handle_ != nullptr) {
        pcap_close(static_cast<pcap_t*>(handle_));
        handle_ = nullptr;
    }
}

bool PcapReader::setFilter(const std::string& filter) {
    if (handle_ == nullptr) return false;
    
    struct bpf_program fp;
    if (pcap_compile(static_cast<pcap_t*>(handle_), &fp, filter.c_str(), 0, PCAP_NETMASK_UNKNOWN) == -1) {
        std::cerr << "Error compiling filter: " << pcap_geterr(static_cast<pcap_t*>(handle_)) << std::endl;
        return false;
    }
    
    if (pcap_setfilter(static_cast<pcap_t*>(handle_), &fp) == -1) {
        std::cerr << "Error setting filter: " << pcap_geterr(static_cast<pcap_t*>(handle_)) << std::endl;
        pcap_freecode(&fp);
        return false;
    }
    
    pcap_freecode(&fp);
    return true;
}

bool PcapReader::parsePacket(const uint8_t* data, uint32_t len, PacketInfo& packet) {
    packet.is_valid = false;
    
    // Minimum Ethernet header size
    if (len < 14) return false;
    
    // Check Ethernet type (0x0800 for IPv4)
    uint16_t eth_type = (data[12] << 8) | data[13];
    if (eth_type != 0x0800) return false;  // Not IPv4
    
    // Skip Ethernet header (14 bytes)
    const uint8_t* ip_data = data + 14;
    uint32_t ip_len = len - 14;
    
    if (ip_len < sizeof(struct ip)) return false;
    
    const struct ip* ip_hdr = reinterpret_cast<const struct ip*>(ip_data);
    
    // Check IP version
    if (ip_hdr->ip_v != 4) return false;
    
    // Get IP header length
    uint8_t ip_hlen = ip_hdr->ip_hl * 4;
    if (ip_hlen < 20 || ip_len < ip_hlen) return false;
    
    packet.src_ip = ip_hdr->ip_src.s_addr;
    packet.dst_ip = ip_hdr->ip_dst.s_addr;
    packet.protocol = ip_hdr->ip_p;
    packet.packet_size = ntohs(ip_hdr->ip_len);
    
    // Parse TCP or UDP header
    const uint8_t* transport_data = ip_data + ip_hlen;
    uint32_t transport_len = ip_len - ip_hlen;
    
    packet.tcp_flags = 0;
    
    if (packet.protocol == 6) {  // TCP
        if (transport_len < sizeof(struct tcphdr)) return false;
        const struct tcphdr* tcp_hdr = reinterpret_cast<const struct tcphdr*>(transport_data);
        packet.src_port = tcp_hdr->th_sport;
        packet.dst_port = tcp_hdr->th_dport;
        packet.tcp_flags = tcp_hdr->th_flags;
    } else if (packet.protocol == 17) {  // UDP
        if (transport_len < sizeof(struct udphdr)) return false;
        const struct udphdr* udp_hdr = reinterpret_cast<const struct udphdr*>(transport_data);
        packet.src_port = udp_hdr->uh_sport;
        packet.dst_port = udp_hdr->uh_dport;
    } else {
        // Other protocols - set ports to 0
        packet.src_port = 0;
        packet.dst_port = 0;
    }
    
    packet.is_valid = true;
    return true;
}

bool PcapReader::readNextPacket(PacketInfo& packet) {
    if (handle_ == nullptr) return false;
    
    struct pcap_pkthdr* header;
    const u_char* data;
    
    int result = pcap_next_ex(static_cast<pcap_t*>(handle_), &header, &data);
    
    if (result == 1) {  // Packet captured
        // Convert timestamp to microseconds
        packet.timestamp_us = header->ts.tv_sec * 1000000ULL + header->ts.tv_usec;
        
        if (parsePacket(data, header->caplen, packet)) {
            packet_count_++;
            return true;
        }
    } else if (result == 0) {
        // Timeout (shouldn't happen with offline files)
        return false;
    } else if (result == -1) {
        // Error
        std::cerr << "Error reading packet: " << pcap_geterr(static_cast<pcap_t*>(handle_)) << std::endl;
        return false;
    } else if (result == -2) {
        // End of file
        return false;
    }
    
    return false;
}

