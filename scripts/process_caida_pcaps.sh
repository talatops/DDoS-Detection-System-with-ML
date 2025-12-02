#!/bin/bash
# Process ALL PCAP files from data directory

DETECTOR="./build/detector"
LOG_DIR="logs"
DATA_DIR="data"

echo "=== Processing ALL PCAP Files in Data Directory ==="
echo ""

# Check if detector exists
if [ ! -f "$DETECTOR" ]; then
    echo "ERROR: Detector not found at $DETECTOR"
    echo "Please rebuild first: cd build && cmake .. && make"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    exit 1
fi

# Collect all PCAP files
PCAP_FILES=()

# 1. Find all files with .pcap extension
echo "Scanning for .pcap files..."
while IFS= read -r file; do
    PCAP_FILES+=("$file")
done < <(find "$DATA_DIR" -type f -name "*.pcap" 2>/dev/null)

# 2. Find all files in directories named PCAP* (caida-ddos2007 files without extension)
echo "Scanning for PCAP directories..."
while IFS= read -r file; do
    # Verify it's actually a PCAP file using file command
    if file "$file" 2>/dev/null | grep -q "pcap capture file"; then
        PCAP_FILES+=("$file")
    fi
done < <(find "$DATA_DIR" -type d -name "PCAP*" -exec find {} -type f \; 2>/dev/null)

total_files=${#PCAP_FILES[@]}
processed=0
failed=0

if [ $total_files -eq 0 ]; then
    echo "WARNING: No PCAP files found in $DATA_DIR"
    exit 1
fi

echo "Found $total_files PCAP file(s) to process"
echo ""

# Process each PCAP file
file_num=0
for pcap_file in "${PCAP_FILES[@]}"; do
    file_num=$((file_num + 1))
    file_name=$(basename "$pcap_file")
    file_dir=$(dirname "$pcap_file")
    
    echo "[$file_num/$total_files] Processing: $file_name"
    echo "  Location: $file_dir"
    
    # Run detector with progress output
    if $DETECTOR --pcap "$pcap_file" --window 1 --batch 128 2>&1 | grep -q "Processed.*packets total"; then
        processed=$((processed + 1))
        echo "  ✓ Success"
    else
        failed=$((failed + 1))
        echo "  ✗ Failed"
    fi
    echo ""
done

echo "════════════════════════════════════════════════════"
echo "=== Processing Summary ==="
echo "════════════════════════════════════════════════════"
echo "Total files found: $total_files"
echo "Successfully processed: $processed"
echo "Failed: $failed"
echo ""
echo "Log files:"
echo "  - Alerts: $LOG_DIR/alerts.csv"
echo "  - Blocking: $LOG_DIR/blocking.csv"
echo "  - Metrics: $LOG_DIR/metrics.csv"
echo "  - Kernel times: $LOG_DIR/kernel_times.csv"
echo ""
echo "To view alerts: tail -f $LOG_DIR/alerts.csv"
echo "To view blocking: tail -f $LOG_DIR/blocking.csv"

