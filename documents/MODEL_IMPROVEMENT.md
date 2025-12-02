# Model Improvement Guide

## Current Issues Identified

Based on the training report (`reports/training_report.txt`), the model has the following issues:

1. **Feature Importance Problem**: Only `total_bytes` has importance (1.0000), all other features are 0.0000
2. **Placeholder Features**: Entropy features are set to 0.0 (placeholders)
3. **Constant Features**: Unique counts and Top-N fractions are constants
4. **Limited Feature Diversity**: Model relies on single feature

## Improvements Made

### 1. Real Entropy Calculation ✅

**Before**: Entropy features were placeholders (0.0)
```python
df_features['src_ip_entropy'] = 0.0  # Placeholder
df_features['dst_ip_entropy'] = 0.0
```

**After**: Real entropy calculated from CSV data
```python
# Group flows into windows (1000 flows per window)
# Calculate Shannon entropy: H = -Σ p_i * log2(p_i)
src_ip_entropy = calculate_entropy(src_ip_counts, window_total)
dst_ip_entropy = calculate_entropy(dst_ip_counts, window_total)
```

### 2. Real Unique Counts ✅

**Before**: Constant values (1)
```python
df_features['unique_src_ips'] = 1  # Constant
```

**After**: Real counts per window
```python
unique_src_ips = len(src_ip_counts)  # Actual unique count
unique_dst_ips = len(dst_ip_counts)
```

### 3. Real Top-N Fractions ✅

**Before**: Constant values (1.0)
```python
df_features['top10_src_ip_fraction'] = 1.0  # Constant
```

**After**: Real fractions calculated
```python
top10_src_fraction = calculate_top_n_fraction(src_ip_counts, window_total, 10)
```

### 4. Additional Features ✅

Added more meaningful features from CSV:
- Flow duration
- Flow bytes/sec
- Flow packets/sec
- Packet size mean/std
- TCP flag counts (SYN, FIN, RST)

### 5. Improved Hyperparameters ✅

**Before**: 
- n_estimators=100
- max_depth=10

**After**:
- n_estimators=200 (more trees for better accuracy)
- max_depth=15 (deeper trees for complex patterns)

### 6. Feature Selection ✅

Enabled feature selection to:
- Remove redundant features
- Focus on top 10 most important features
- Improve model generalization

## Retraining Process

### Step 1: Retrain with Real Entropy

```bash
# Retrain model with improved features
python3 src/ml/train_ml.py
```

**Expected Improvements**:
- ✅ Multiple features with non-zero importance
- ✅ Entropy features contributing to detection
- ✅ Better feature diversity
- ✅ Potentially improved accuracy

### Step 2: Compare Results

Compare old vs new model:
```bash
# Old model report
cat reports/training_report.txt

# New model report (after retraining)
cat reports/training_report.txt
```

**Key Metrics to Compare**:
- Feature importance distribution
- Test accuracy
- ROC AUC
- F1 Score
- Precision/Recall

### Step 3: Analyze Feature Importance

After retraining, check feature importance:
```bash
python3 -c "
import joblib
import numpy as np
model = joblib.load('models/rf_model.joblib')
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
print('Top 10 Features:')
for i in range(10):
    print(f'{i+1}. Importance: {importances[indices[i]]:.4f}')
"
```

## Expected Outcomes

### Before (Placeholder Features)
- Only `total_bytes` matters (importance: 1.0000)
- All entropy features: 0.0000
- Model relies on single feature
- Limited generalization

### After (Real Entropy)
- Multiple features contribute
- Entropy features have non-zero importance
- Better feature diversity
- Improved model robustness
- Better detection of DDoS patterns

## Validation

After retraining, validate improvements:

1. **Feature Importance Check**:
   - At least 3-5 features should have importance > 0.1
   - Entropy features should have importance > 0.05

2. **Performance Metrics**:
   - Accuracy should be similar or better
   - ROC AUC should be similar or better
   - F1 Score should be similar or better

3. **Feature Diversity**:
   - No single feature should dominate (>0.8 importance)
   - Entropy features should contribute meaningfully

## Next Steps

1. ✅ Retrain model with real entropy
2. ✅ Compare old vs new results
3. ✅ Analyze feature importance
4. ⏳ If needed, further tune hyperparameters
5. ⏳ Consider ensemble methods
6. ⏳ Add more features if needed

## Notes

- **Window Size**: Currently using 1000 flows per window for entropy calculation
- **Entropy Formula**: Shannon entropy H = -Σ p_i * log2(p_i)
- **Feature Selection**: Enabled to select top 10 features
- **Hyperparameters**: Increased to 200 estimators, max_depth=15

---

**Status**: Ready for retraining with improved features! 🚀

