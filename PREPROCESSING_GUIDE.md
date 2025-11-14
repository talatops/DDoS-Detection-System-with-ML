# Preprocessing Guide

## Two Types of Preprocessing

### 1. **Feature Preprocessing** (`src/ml/preprocessor.py`) ✅ **ALREADY INTEGRATED**

**What it does:**
- Removes outliers (clips to 1st-99th percentiles)
- Optional feature scaling (disabled for Random Forest)
- Optional feature selection (disabled by default)

**Do you need to run it separately?**
- **NO** - It's already integrated into `train_ml.py` (lines 204-216)
- It runs automatically during training
- The preprocessor is saved to `models/preprocessor.joblib` for later use

**How it works:**
```python
# In train_ml.py (already there)
preprocessor = FeaturePreprocessor(
    use_scaling=False,
    use_outlier_removal=True,  # ✅ Enabled
    use_feature_selection=False
)
X_processed = preprocessor.fit_transform(X, y)  # Runs automatically
preprocessor.save("models/preprocessor.joblib")  # Saved for inference
```

---

### 2. **Data Preparation** (`scripts/prepare_data.py`) ⚠️ **OPTIONAL**

**What it does:**
- Combines multiple CSV files into one file
- Adds source file information
- Creates `data/processed/combined_features.csv`

**Do you need to run it?**
- **NO** - `train_ml.py` can read CSV files directly
- It's just a convenience script
- Only useful if you want a single combined file

**When to use it:**
- If you want to pre-process data once and reuse it
- If you want to inspect combined data before training
- Otherwise, skip it - `train_ml.py` handles CSV files directly

---

## What Happens During Training

### Automatic Flow:

```
1. train_ml.py loads CSV files directly
   ↓
2. Extracts features from CSV (extract_features_from_csv)
   ↓
3. FeaturePreprocessor runs automatically:
   - Removes outliers
   - (Optional) Scales features
   - (Optional) Selects features
   ↓
4. Preprocessor saved to models/preprocessor.joblib
   ↓
5. Model trained on preprocessed features
   ↓
6. Model saved to models/rf_model.joblib
```

---

## Answer: Do You Need to Run Preprocessing First?

### ✅ **NO - Just run training directly:**

```bash
# This is all you need:
python3 src/ml/train_ml.py
```

**What happens:**
1. CSV files are loaded automatically
2. Features are extracted automatically
3. Preprocessing runs automatically (outlier removal)
4. Preprocessor is saved automatically
5. Model is trained and saved

### ⚠️ **Optional: Run prepare_data.py first (only if you want)**

```bash
# Optional step (not required):
python3 scripts/prepare_data.py

# Then train (but train_ml.py can work without this):
python3 src/ml/train_ml.py
```

---

## Summary

| Script | Required? | What It Does | When to Use |
|--------|-----------|--------------|-------------|
| `train_ml.py` | ✅ **YES** | Trains model + runs preprocessing | Always |
| `preprocessor.py` | ✅ **Integrated** | Feature preprocessing class | Used automatically |
| `prepare_data.py` | ❌ **NO** | Combines CSV files | Optional convenience |

---

## Quick Start

**Just run this:**
```bash
cd /home/talatfaheem/PDC/project
python3 src/ml/train_ml.py
```

**That's it!** Preprocessing happens automatically. ✅

