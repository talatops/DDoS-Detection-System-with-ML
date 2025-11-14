# Dataset Breakdown - CAIDA DDoS 2007

## CSV Files Used in Training

### Folder 1: CSV-01-12/01-12 (9 files)
These contain **DrDoS (Distributed Reflection DDoS)** attacks:

1. `DrDoS_NetBIOS.csv`
2. `DrDoS_SSDP.csv`
3. `Syn.csv`
4. `DrDoS_LDAP.csv`
5. `DrDoS_DNS.csv`
6. `DrDoS_SNMP.csv`
7. `DrDoS_UDP.csv`
8. `DrDoS_NTP.csv`
9. `DrDoS_MSSQL.csv`

**Attack Type:** Distributed Reflection attacks (amplification attacks)

---

### Folder 2: CSV-03-11/03-11 (7 files)
These contain **basic DDoS attacks**:

1. `NetBIOS.csv`
2. `UDP.csv`
3. `Syn.csv`
4. `Portmap.csv`
5. `UDPLag.csv`
6. `LDAP.csv`
7. `MSSQL.csv`

**Attack Type:** Direct DDoS attacks (flooding attacks)

---

## Training Data Summary

**Total CSV Files:** 16 files
- CSV-01-12: 9 files (DrDoS attacks)
- CSV-03-11: 7 files (Basic attacks)

**Total Rows Loaded:** 800,000 rows (50,000 per file)
- Attack samples: 778,115
- Benign samples: 21,885

**Note:** Some files have the same name (e.g., `Syn.csv` appears in both folders) but contain different attack patterns.

---

## Why Both Folders?

1. **Diversity**: Different attack types (reflection vs direct flooding)
2. **Coverage**: More attack patterns = better model generalization
3. **Balance**: Mix of attack types improves detection accuracy

---

## File Sizes

**CSV-03-11 files are much larger:**
- Some files are 2GB+ (e.g., `MSSQL.csv` = 2.3GB)
- CSV-01-12 files are smaller

**Current Training:**
- Loads 50,000 rows per file (for speed)
- Can increase if needed for better accuracy

---

## Verification

The training script uses `rglob("*.csv")` which automatically finds files in **both folders**:

```python
csv_dir_path = Path("data/caida-ddos2007")
for csv_file in csv_dir_path.rglob("*.csv"):  # Finds files in all subdirectories
    # Loads from both CSV-01-12 and CSV-03-11
```

✅ **Both folders are being used automatically!**

