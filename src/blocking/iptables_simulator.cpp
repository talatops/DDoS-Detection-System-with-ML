#include "iptables_simulator.h"
#include <sstream>
#include <iostream>
#include <algorithm>
#include <arpa/inet.h>
#include <cstdlib>

IptablesSimulator::IptablesSimulator(const std::string& rules_file)
    : rules_file_(rules_file) {
    loadRulesFromFile();
}

IptablesSimulator::~IptablesSimulator() {
    saveRulesToFile();
}

std::string IptablesSimulator::ipUintToString(uint32_t ip) const {
    struct in_addr addr;
    addr.s_addr = ip;
    return std::string(inet_ntoa(addr));
}

std::string IptablesSimulator::generateRule(uint32_t ip, const std::string& chain) {
    std::ostringstream oss;
    oss << "iptables -A " << chain << " -s " << ipUintToString(ip) << " -j DROP";
    return oss.str();
}

std::string IptablesSimulator::generateRule(const std::string& ip_str, const std::string& chain) {
    std::ostringstream oss;
    oss << "iptables -A " << chain << " -s " << ip_str << " -j DROP";
    return oss.str();
}

bool IptablesSimulator::addRule(uint32_t ip) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string rule = generateRule(ip);
    
    // Check if rule already exists
    if (std::find(rules_.begin(), rules_.end(), rule) == rules_.end()) {
        rules_.push_back(rule);
        return true;
    }
    
    return false;
}

bool IptablesSimulator::addRule(const std::string& ip_str) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string rule = generateRule(ip_str);
    
    if (std::find(rules_.begin(), rules_.end(), rule) == rules_.end()) {
        rules_.push_back(rule);
        return true;
    }
    
    return false;
}

bool IptablesSimulator::removeRule(uint32_t ip) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string rule = generateRule(ip);
    
    auto it = std::find(rules_.begin(), rules_.end(), rule);
    if (it != rules_.end()) {
        rules_.erase(it);
        return true;
    }
    
    return false;
}

bool IptablesSimulator::removeRule(const std::string& ip_str) {
    std::lock_guard<std::mutex> lock(mutex_);
    std::string rule = generateRule(ip_str);
    
    auto it = std::find(rules_.begin(), rules_.end(), rule);
    if (it != rules_.end()) {
        rules_.erase(it);
        return true;
    }
    
    return false;
}

bool IptablesSimulator::generateRulesFromRTBH(RTBHController* rtbh_controller) {
    if (!rtbh_controller) return false;
    
    std::lock_guard<std::mutex> lock(mutex_);
    rules_.clear();
    
    std::vector<uint32_t> blackholed_ips = rtbh_controller->getBlackholedIPs();
    for (uint32_t ip : blackholed_ips) {
        rules_.push_back(generateRule(ip));
    }
    
    return true;
}

bool IptablesSimulator::executeRules(bool dry_run) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (dry_run) {
        std::cout << "DRY RUN - Would execute " << rules_.size() << " iptables rules:" << std::endl;
        for (const auto& rule : rules_) {
            std::cout << "  " << rule << std::endl;
        }
        return true;
    }
    
    // Safety check: only execute if explicitly enabled
    // In production, add more safety checks
    std::cout << "WARNING: Executing iptables rules requires root privileges" << std::endl;
    std::cout << "This feature is disabled by default for safety" << std::endl;
    
    // Uncomment below to enable (use with caution!)
    /*
    for (const auto& rule : rules_) {
        std::string cmd = "sudo " + rule;
        int result = system(cmd.c_str());
        if (result != 0) {
            std::cerr << "Failed to execute: " << rule << std::endl;
            return false;
        }
    }
    */
    
    return false;  // Disabled by default
}

std::vector<std::string> IptablesSimulator::getRules() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return rules_;
}

void IptablesSimulator::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    rules_.clear();
}

bool IptablesSimulator::saveRulesToFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ofstream file(rules_file_);
    if (!file.is_open()) {
        std::cerr << "Failed to open rules file for writing: " << rules_file_ << std::endl;
        return false;
    }
    
    for (const auto& rule : rules_) {
        file << rule << "\n";
    }
    
    file.close();
    return true;
}

bool IptablesSimulator::loadRulesFromFile() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ifstream file(rules_file_);
    if (!file.is_open()) {
        // File doesn't exist yet, that's okay
        return true;
    }
    
    rules_.clear();
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            rules_.push_back(line);
        }
    }
    
    file.close();
    return true;
}

