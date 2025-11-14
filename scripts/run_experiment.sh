#!/bin/bash
# Experiment automation script for DDoS detection system

set -e

# Default values
PCAP_FILE=""
PPS=200000
WINDOW_SECS=1
BATCH_SIZE=128
MODEL="rf"
OUTPUT_DIR="logs/exp_$(date +%Y%m%d_%H%M%S)"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --pcap)
            PCAP_FILE="$2"
            shift 2
            ;;
        --pps)
            PPS="$2"
            shift 2
            ;;
        --window)
            WINDOW_SECS="$2"
            shift 2
            ;;
        --batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [ -z "$PCAP_FILE" ]; then
    echo "Usage: $0 --pcap <file> [--pps <rate>] [--window <secs>] [--batch <size>] [--model <model>]"
    exit 1
fi

echo "=== Starting Experiment ==="
echo "PCAP: $PCAP_FILE"
echo "PPS: $PPS"
echo "Window: ${WINDOW_SECS}s"
echo "Batch: $BATCH_SIZE"
echo "Model: $MODEL"
echo "Output: $OUTPUT_DIR"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Start metrics collector
echo "Starting metrics collector..."
python3 -c "
import sys
sys.path.insert(0, 'src')
from utils.metrics_collector import MetricsCollector
import time
collector = MetricsCollector()
collector.start('$OUTPUT_DIR')
time.sleep(3600)  # Run for 1 hour
" &
METRICS_PID=$!

# Start detector (would be the compiled C++ binary)
echo "Starting detector..."
# ./bin/detector --pcap "$PCAP_FILE" --window "$WINDOW_SECS" --batch "$BATCH_SIZE" --model "$MODEL" --output "$OUTPUT_DIR" &
# DETECTOR_PID=$!

# Start packet replay
echo "Starting packet replay..."
tcpreplay --pps="$PPS" --intf1=lo "$PCAP_FILE" &
REPLAY_PID=$!

# Wait for replay to finish
wait $REPLAY_PID

# Stop detector and metrics collector
echo "Stopping components..."
kill $METRICS_PID 2>/dev/null || true
# kill $DETECTOR_PID 2>/dev/null || true

# Generate plots
echo "Generating plots..."
python3 scripts/plot_results.py "$OUTPUT_DIR"

echo "Experiment complete. Results in $OUTPUT_DIR"

