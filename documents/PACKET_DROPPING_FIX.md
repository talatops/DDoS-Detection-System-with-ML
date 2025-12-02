# Packet Dropping Fix

## Problem
The blocking system was logging blocking events but not actually dropping packets. The `blocking.csv` showed `dropped_packets=0` for most entries because:
1. Packets were not being filtered during the read loop
2. Only the initial window's packets were counted as "dropped" when an IP was first added
3. Subsequent packets from blackholed IPs were still being processed

## Solution
Updated `src/main.cpp` to:

1. **Filter packets during read loop**: Added `pcap_filter.shouldDrop()` check right after reading each packet
   - If packet's source or destination IP is blackholed, skip it (don't add to window)
   - Track which IP caused the drop

2. **Track dropped packets per IP**: Maintain a map of dropped packets per IP
   - Increment counter when a packet is dropped
   - Reset counters periodically when logging

3. **Periodic blocking log**: Log dropped packets every second
   - Shows actual dropped packet counts per IP
   - Format: `timestamp_ms,ip,impacted_packets,dropped_packets`
   - `impacted_packets=0` for periodic logs (only counts actual drops)
   - `dropped_packets` shows real dropped count

4. **Initial blocking log**: When IP is first added to blackhole
   - `impacted_packets` = packets in that window (already processed)
   - `dropped_packets` = 0 (packets were already processed, future packets will be dropped)

5. **Summary output**: Shows total dropped packets at end of processing

## Changes Made
- Added `#include <unordered_map>` for tracking
- Added packet filtering in read loop (line ~348)
- Added per-IP dropped packet tracking
- Added periodic blocking log (every 1 second)
- Updated initial blocking log to show `dropped_packets=0`
- Added final summary of dropped packets

## Expected Behavior
After rebuild:
- Packets from blackholed IPs will be dropped (not processed)
- `blocking.csv` will show actual dropped packet counts
- Periodic logs will show continuous dropping activity
- Console will show total dropped packets at end

## Rebuild Required
```bash
cd build && rm -rf * && cmake .. && make -j4
```
