# Rebuild Instructions and PCAP Processing Guide

## Changes Made

### 1. Blocking System (src/main.cpp)
- Added IP blocking when attacks are detected
- IPs are automatically added to blackhole list
- Blocking events logged to logs/blocking.csv

### 2. Logger Append Mode (src/utils/logger.cpp)  
- Changed to append mode so multiple runs don't overwrite logs
- Headers only written for new/empty files

## Rebuild Required

The detector needs to be rebuilt to activate these changes:

```bash
cd build
make clean
cmake ..
make detector
```

If build fails, try:
```bash
cd build
rm -rf *
cmake ..
make -j4
```

## PCAP Files to Process

### Already Processed (cic-ddos2019):
1. ✅ ddostrace.20070804_145436.pcap (408MB) - 55 windows, 9.4M packets
2. ✅ ddostrace.20070804_141436.pcap (1.2GB) - 300 windows, 26.7M packets  
3. ✅ ddostrace.20070804_142936.pcap (2.0GB) - 300 windows, 45.9M packets

### Available in caida-ddos2007:
- PCAP-01-12: 250 files (~191MB each, no .pcap extension)
- PCAP-03-11: 146 files (~191MB each, no .pcap extension)

These are valid PCAP files and can be processed directly.

## Processing Script

See process_all_pcaps.sh for batch processing.
