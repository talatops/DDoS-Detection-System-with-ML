# Next Steps After Rebuild

## ✅ Rebuild Complete!

The detector has been successfully rebuilt with:
- ✅ Packet dropping functionality (blackholed IPs are now filtered)
- ✅ Logger append mode (logs won't be overwritten)
- ✅ GPU kernel acceleration (entropy calculation)

---

## 🎯 What's Next?

### Option 1: Test the Updated Features (Recommended First)

Test the new packet dropping functionality on a small file:

```bash
# Test on a single PCAP file
./build/detector --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap --window 1 --batch 128
```

**What to check:**
- Console output should show: `Dropped X packets (blackholed IPs)`
- `logs/blocking.csv` should show actual dropped packet counts (not just 0)
- `logs/alerts.csv` should append (not overwrite) if you run multiple times

### Option 2: Process All PCAP Files

Run the updated script to process all 399 PCAP files:

```bash
./process_caida_pcaps.sh
```

**What this will do:**
- Process 3 files from `data/cic-ddos2019/` (with .pcap extension)
- Process 396 files from `data/caida-ddos2007/PCAP-*` directories (no extension)
- Total: **399 PCAP files**
- Estimated time: Several hours (depends on file sizes)

**Note:** This is a long-running process. Consider running in `screen` or `tmux`:

```bash
# Using screen
screen -S pcap_processing
./process_caida_pcaps.sh
# Press Ctrl+A then D to detach
# Reattach later with: screen -r pcap_processing

# Or using tmux
tmux new -s pcap_processing
./process_caida_pcaps.sh
# Press Ctrl+B then D to detach
# Reattach later with: tmux attach -t pcap_processing
```

### Option 3: Process Specific Directories

If you want to process specific directories:

```bash
# Process only cic-ddos2019 files
for file in data/cic-ddos2019/*.pcap; do
    ./build/detector --pcap "$file" --window 1 --batch 128
done

# Process only caida-ddos2007 files
for file in data/caida-ddos2007/PCAP-01-12/*; do
    ./build/detector --pcap "$file" --window 1 --batch 128
done
```

---

## 📊 Monitoring Progress

### While Processing:

**View alerts in real-time:**
```bash
tail -f logs/alerts.csv
```

**View blocking events:**
```bash
tail -f logs/blocking.csv
```

**View metrics:**
```bash
tail -f logs/metrics.csv
```

**View kernel times (GPU performance):**
```bash
tail -f logs/kernel_times.csv
```

### After Processing:

**Count total alerts:**
```bash
echo "Total alerts: $(($(wc -l < logs/alerts.csv) - 1))"
```

**Count total blocked IPs:**
```bash
echo "Total unique blocked IPs: $(tail -n +2 logs/blocking.csv | cut -d',' -f2 | sort -u | wc -l)"
```

**Count total dropped packets:**
```bash
echo "Total dropped packets: $(tail -n +2 logs/blocking.csv | awk -F',' '{sum+=$4} END {print sum}')"
```

---

## 🔍 Verify New Features Work

### 1. Packet Dropping Verification

After processing a file, check `logs/blocking.csv`:
- Should have entries with `dropped_packets > 0` (not all zeros)
- Each IP should have periodic logs showing dropped packets

Example:
```csv
timestamp_ms,ip,impacted_packets,dropped_packets
1543686604339,172.16.0.5,69419,0          # Initial block (packets already processed)
1543686605339,172.16.0.5,0,1250          # Periodic log (actual drops)
1543686606339,172.16.0.5,0,1180          # Periodic log (actual drops)
```

### 2. Logger Append Mode Verification

Run the detector twice on the same file:
```bash
./build/detector --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap --window 1 --batch 128
./build/detector --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap --window 1 --batch 128
```

Check `logs/alerts.csv`:
- Should have alerts from BOTH runs (not just the second)
- Alert count should be approximately double

### 3. GPU Kernel Verification

Check `logs/kernel_times.csv`:
- Should have entries for `compute_multi_entropy`
- Execution times should be in milliseconds (0.1-5ms typical)

---

## 📈 Expected Results

### From cic-ddos2019 files (3 files):
- **File 1** (408MB): ~55 windows, ~9.4M packets
- **File 2** (1.2GB): ~300 windows, ~26.7M packets  
- **File 3** (2.0GB): ~300 windows, ~45.9M packets

### From caida-ddos2007 files (396 files):
- Each file: ~191MB, ~250 windows, ~20-30M packets each
- Total: ~99GB, ~99,000 windows, ~8-12 billion packets

**Note:** Processing all 399 files will take significant time and generate large log files.

---

## 🚀 Quick Start Commands

```bash
# 1. Test on one file first
./build/detector --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap --window 1 --batch 128

# 2. Check results
echo "Alerts: $(($(wc -l < logs/alerts.csv) - 1))"
echo "Blocked IPs: $(tail -n +2 logs/blocking.csv | cut -d',' -f2 | sort -u | wc -l)"
echo "Dropped packets: $(tail -n +2 logs/blocking.csv | awk -F',' '{sum+=$4} END {print sum}')"

# 3. If everything looks good, process all files
./process_caida_pcaps.sh
```

---

## ⚠️ Important Notes

1. **Disk Space**: Processing all files will generate large log files (potentially GBs)
   - Monitor disk space: `df -h`

2. **Time**: Processing 399 files will take many hours
   - Use `screen` or `tmux` to run in background

3. **Memory**: Large files may use significant RAM
   - Monitor with: `free -h`

4. **Log Files**: Logs are now in append mode
   - Old logs won't be overwritten
   - Consider backing up or archiving old logs before processing

---

## 📝 Summary

✅ **Rebuild complete** - All new features are active
✅ **Script updated** - Will process all 399 PCAP files
✅ **Ready to test** - Start with a single file to verify

**Next action:** Test on one file, then decide whether to process all files or specific subsets.

