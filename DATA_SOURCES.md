# Data Sources Used in DDoS Detection System

## Current Training: CSV Files Only ✅

**What was used:**
- ✅ **16 CSV files** from `data/caida-ddos2007/`
- ✅ **800,000 rows** of flow-level data
- ❌ **PCAP files NOT used** for training

**Why CSV files for training?**
- CSV files contain **pre-aggregated flow statistics**
- Easier to extract features quickly
- Faster training (no need to parse packets)
- Contains labels (attack/benign)

---

## PCAP Files: Used Later for Runtime Detection

**When PCAP files are used:**
- ✅ **During runtime detection** (when processing live/recorded traffic)
- ✅ **For testing** the full pipeline
- ✅ **For GPU acceleration** (entropy calculation, feature extraction)

**How PCAP files are used:**
```
PCAP File → libpcap → Parse Packets → Create Windows → GPU Processing → Detection
```

---

## Data Flow Summary

### Training Phase (What Just Happened):
```
CSV Files (flow statistics)
    ↓
Feature Extraction (from CSV columns)
    ↓
Preprocessing (outlier removal)
    ↓
Random Forest Training (CPU)
    ↓
Model Saved: models/rf_model.joblib
```

### Runtime Detection Phase (Future):
```
PCAP Files (raw packets)
    ↓
Packet Parsing (libpcap)
    ↓
Windowing (1s, 5s windows)
    ↓
GPU Feature Extraction (OpenCL) ← HEAVY GPU USAGE
    ↓
ML Inference (CPU, uses trained model)
    ↓
Detection & Alerts
```

---

## What Your Training Used

From your output:
```
Found 16 CSV files
Loaded: NetBIOS.csv, UDP.csv, Syn.csv, Portmap.csv, etc.
Total: 800,000 rows
  - 778,115 attacks
  - 21,885 benign
```

**Data Sources Used:**
- ✅ `data/caida-ddos2007/CSV-01-12/01-12/*.csv` (9 files - mostly DrDoS attacks)
- ✅ `data/caida-ddos2007/CSV-03-11/03-11/*.csv` (7 files - basic attack types)
- **Total: 16 CSV files from both folders**

**NOT Used:** PCAP files (those are for runtime detection)

---

## Why This Approach?

### ✅ Advantages of CSV for Training:
1. **Fast**: Pre-aggregated data loads quickly
2. **Labeled**: CSV files have attack/benign labels
3. **Simple**: Direct feature extraction from columns
4. **Efficient**: No packet parsing needed

### ✅ Advantages of PCAP for Runtime:
1. **Real-time**: Process live traffic
2. **GPU acceleration**: Parallel packet processing
3. **Detailed**: Full packet-level information
4. **Realistic**: Actual network traffic format

---

## Summary

| Phase | Data Source | Purpose | GPU Usage |
|-------|-------------|---------|-----------|
| **Training** | CSV files | Train ML model | ❌ No (CPU only) |
| **Runtime Detection** | PCAP files | Detect attacks | ✅ Yes (Heavy GPU) |

**Your training:** ✅ Used CSV files only (correct!)
**Next step:** Use PCAP files for runtime detection testing

---

## Next Steps

1. ✅ **Training Complete** - Model saved to `models/rf_model.joblib`
2. **Test with PCAP files:**
   ```bash
   # After building the C++ detector
   ./bin/detector --pcap data/cic-ddos2019/ddostrace.20070804_145436.pcap \
                  --model models/rf_model.joblib
   ```
3. **GPU will be used** during PCAP processing (entropy, feature extraction)

