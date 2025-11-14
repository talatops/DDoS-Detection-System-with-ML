#include "rtbh_controller.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <arpa/inet.h>

RTBHController::RTBHController(const std::string& blackhole_file)
    : blackhole_file_(blackhole_file) {
    loadFromFile();
}

RTBHController::~RTBHController() {
    saveToFile();
}

uint32_t RTBHController::ipStringToUint(const std::string& ip_str) const {
    struct in_addr addr;
    if (inet_aton(ip_str.c_str(), &addr) == 0) {
        return 0;
    }
    return addr.s_addr;
}

std::string RTBHController::ipUintToString(uint32_t ip) const {
    struct in_addr addr;
    addr.s_addr = ip;
    return std::string(inet_ntoa(addr));
}

bool RTBHController::addIP(uint32_t ip) {
    std::lock_guard<std::mutex> lock(mutex_);
    blackhole_list_.insert(ip);
    return true;
}

bool RTBHController::addIP(const std::string& ip_str) {
    uint32_t ip = ipStringToUint(ip_str);
    if (ip == 0) return false;
    return addIP(ip);
}

bool RTBHController::removeIP(uint32_t ip) {
    std::lock_guard<std::mutex> lock(mutex_);
    return blackhole_list_.erase(ip) > 0;
}

bool RTBHController::removeIP(const std::string& ip_str) {
    uint32_t ip = ipStringToUint(ip_str);
    if (ip == 0) return false;
    return removeIP(ip);
}

bool RTBHController::isBlackholed(uint32_t ip) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return blackhole_list_.find(ip) != blackhole_list_.end();
}

bool RTBHController::isBlackholed(const std::string& ip_str) const {
    uint32_t ip = ipStringToUint(ip_str);
    if (ip == 0) return false;
    return isBlackholed(ip);
}

std::vector<uint32_t> RTBHController::getBlackholedIPs() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return std::vector<uint32_t>(blackhole_list_.begin(), blackhole_list_.end());
}

bool RTBHController::saveToFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(blackhole_file_);
    if (!file.is_open()) {
        std::cerr << "Failed to open blackhole file for writing: " << blackhole_file_ << std::endl;
        return false;
    }
    
    file << "{\n";
    file << "  \"blackhole_ips\": [\n";
    
    bool first = true;
    for (uint32_t ip : blackhole_list_) {
        if (!first) file << ",\n";
        file << "    \"" << ipUintToString(ip) << "\"";
        first = false;
    }
    
    file << "\n  ]\n";
    file << "}\n";
    
    file.close();
    return true;
}

bool RTBHController::loadFromFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream file(blackhole_file_);
    if (!file.is_open()) {
        // File doesn't exist yet, that's okay
        return true;
    }
    
    // Simple JSON parsing (for production, use a proper JSON library)
    std::string line;
    bool in_array = false;
    
    while (std::getline(file, line)) {
        // Look for IP addresses in quotes
        size_t start = line.find('"');
        while (start != std::string::npos) {
            size_t end = line.find('"', start + 1);
            if (end != std::string::npos) {
                std::string ip_str = line.substr(start + 1, end - start - 1);
                uint32_t ip = ipStringToUint(ip_str);
                if (ip != 0) {
                    blackhole_list_.insert(ip);
                }
            }
            start = line.find('"', end + 1);
        }
    }
    
    file.close();
    return true;
}

void RTBHController::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    blackhole_list_.clear();
}

