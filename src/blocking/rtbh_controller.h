#ifndef RTBH_CONTROLLER_H
#define RTBH_CONTROLLER_H

#include <string>
#include <set>
#include <vector>
#include <cstdint>
#include <fstream>
#include <mutex>

class RTBHController {
public:
    RTBHController(const std::string& blackhole_file = "blackhole.json");
    ~RTBHController();
    
    // Add IP to blackhole list
    bool addIP(uint32_t ip);
    bool addIP(const std::string& ip_str);
    
    // Remove IP from blackhole list
    bool removeIP(uint32_t ip);
    bool removeIP(const std::string& ip_str);
    
    // Check if IP is blackholed
    bool isBlackholed(uint32_t ip) const;
    bool isBlackholed(const std::string& ip_str) const;
    
    // Get all blackholed IPs
    std::vector<uint32_t> getBlackholedIPs() const;
    
    // Save blackhole list to file
    bool saveToFile();
    
    // Load blackhole list from file
    bool loadFromFile();
    
    // Clear all blackholes
    void clear();
    
    // Get count
    size_t getCount() const { return blackhole_list_.size(); }

private:
    std::string blackhole_file_;
    std::set<uint32_t> blackhole_list_;
    mutable std::mutex mutex_;
    
    uint32_t ipStringToUint(const std::string& ip_str) const;
    std::string ipUintToString(uint32_t ip) const;
};

#endif // RTBH_CONTROLLER_H

