#ifndef IPTABLES_SIMULATOR_H
#define IPTABLES_SIMULATOR_H

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>
#include <mutex>
#include "blocking/rtbh_controller.h"

class IptablesSimulator {
public:
    IptablesSimulator(const std::string& rules_file = "iptables_rules.txt");
    ~IptablesSimulator();
    
    // Generate iptables rule for IP
    std::string generateRule(uint32_t ip, const std::string& chain = "INPUT");
    std::string generateRule(const std::string& ip_str, const std::string& chain = "INPUT");
    
    // Add rule to file
    bool addRule(uint32_t ip);
    bool addRule(const std::string& ip_str);
    
    // Remove rule
    bool removeRule(uint32_t ip);
    bool removeRule(const std::string& ip_str);
    
    // Generate rules from RTBH controller
    bool generateRulesFromRTBH(RTBHController* rtbh_controller);
    
    // Execute rules (requires sudo, with safety checks)
    bool executeRules(bool dry_run = true);
    
    // Get all rules
    std::vector<std::string> getRules() const;
    
    // Clear all rules
    void clear();

private:
    std::string rules_file_;
    std::vector<std::string> rules_;
    mutable std::mutex mutex_;
    
    bool saveRulesToFile();
    bool loadRulesFromFile();
    std::string ipUintToString(uint32_t ip) const;
};

#endif // IPTABLES_SIMULATOR_H

