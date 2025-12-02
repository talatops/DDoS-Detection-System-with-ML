#!/usr/bin/env python3
"""
Train Random Forest model for DDoS detection.
Uses CIC-DDoS2019 and CAIDA datasets.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.ml.feature_extractor import FeatureExtractor
from src.ml.preprocessor import FeaturePreprocessor

def load_training_data(csv_dir="data/caida-ddos2007", max_rows=None):
    """Load training data from CSV files."""
    print("Loading training data...")
    
    all_data = []
    csv_files = []
    
    # Find all CSV files
    csv_dir_path = Path(csv_dir)
    for csv_file in csv_dir_path.rglob("*.csv"):
        csv_files.append(csv_file)
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load each CSV file
    for csv_file in csv_files:
        try:
            print(f"Loading {csv_file.name}...")
            nrows = max_rows if max_rows and max_rows > 0 else None
            if nrows:
                print(f"  (limiting to first {nrows} rows)")
            df = pd.read_csv(csv_file, nrows=nrows, low_memory=False)
            
            # Check if Label column exists (case-insensitive, handle spaces)
            label_col = None
            for col in df.columns:
                if col.strip().lower() == 'label':
                    label_col = col
                    break
            
            if label_col:
                # Map labels: BENIGN -> 0, attack types -> 1
                df['label'] = (df[label_col] != 'BENIGN').astype(int)
                all_data.append(df)
                print(f"  Loaded {len(df)} rows, {df['label'].sum()} attacks, {len(df[df['label']==0])} benign")
            else:
                # If no Label column, infer from filename (attack files)
                attack_keywords = ['ddos', 'drdos', 'syn', 'udp', 'ldap', 'mssql', 'netbios', 'ssdp', 'dns', 'snmp', 'ntp', 'portmap']
                filename_lower = csv_file.name.lower()
                is_attack = any(keyword in filename_lower for keyword in attack_keywords)
                df['label'] = 1 if is_attack else 0
                all_data.append(df)
                print(f"  Loaded {len(df)} rows (no Label column, inferred: {'ATTACK' if is_attack else 'BENIGN'})")
        except Exception as e:
            print(f"  Error loading {csv_file}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_data:
        print("No data loaded!")
        return None, None
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal combined: {len(combined_df)} rows")
    print(f"Attack samples: {combined_df['label'].sum()}")
    print(f"Benign samples: {(combined_df['label'] == 0).sum()}")
    
    return combined_df, csv_files

def calculate_entropy(counts_dict, total):
    """Calculate Shannon entropy: H = -Σ p_i * log2(p_i)"""
    if total == 0:
        return 0.0
    
    import math
    entropy = 0.0
    for count in counts_dict.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def calculate_top_n_fraction(counts_dict, total, n=10):
    """Calculate fraction of traffic from top N sources."""
    if total == 0:
        return 1.0
    
    sorted_counts = sorted(counts_dict.values(), reverse=True)
    top_n_sum = sum(sorted_counts[:min(n, len(sorted_counts))])
    return top_n_sum / total

def extract_features_from_csv(df):
    """Extract features from CSV flow data with REAL entropy calculation."""
    print("\nExtracting features with REAL entropy calculation...")
    
    # Initialize DataFrame with same index as input
    df_features = pd.DataFrame(index=df.index)
    
    # Helper function to safely get column with default
    def safe_get(col_name, default=0):
        if col_name in df.columns:
            return df[col_name].fillna(default)
        return pd.Series([default] * len(df), index=df.index)
    
    # Basic counts
    df_features['total_packets'] = safe_get('Total Fwd Packets', 0) + safe_get('Total Backward Packets', 0)
    df_features['total_bytes'] = safe_get('Total Length of Fwd Packets', 0) + safe_get('Total Length of Bwd Packets', 0)
    
    # Flow duration and rates
    df_features['flow_duration'] = safe_get(' Flow Duration', 0)
    df_features['flow_bytes_per_sec'] = safe_get('Flow Bytes/s', 0)
    df_features['flow_packets_per_sec'] = safe_get(' Flow Packets/s', 0)
    
    # Packet size statistics
    fwd_mean = safe_get(' Fwd Packet Length Mean', 0)
    fwd_std = safe_get(' Fwd Packet Length Std', 0)
    bwd_mean = safe_get(' Bwd Packet Length Mean', 0)
    bwd_std = safe_get('Bwd Packet Length Std', 0)
    
    df_features['avg_packet_size'] = df_features['total_bytes'] / df_features['total_packets'].replace(0, 1)
    df_features['packet_size_mean'] = (fwd_mean + bwd_mean) / 2
    df_features['packet_size_std'] = (fwd_std + bwd_std) / 2
    
    # TCP flags (if available)
    df_features['syn_flag_count'] = safe_get('SYN Flag Count', 0)
    df_features['fin_flag_count'] = safe_get('FIN Flag Count', 0)
    df_features['rst_flag_count'] = safe_get('RST Flag Count', 0)
    
    # Calculate REAL entropy and unique counts per window
    # Group by time windows (using Flow Duration or Timestamp if available)
    print("  Calculating entropy per window...")
    
    # Use Flow Duration to create windows (or use row numbers as proxy)
    window_size = 1000  # Process in windows of 1000 flows
    n_windows = len(df) // window_size
    
    # Initialize entropy columns
    df_features['src_ip_entropy'] = 0.0
    df_features['dst_ip_entropy'] = 0.0
    df_features['src_port_entropy'] = 0.0
    df_features['dst_port_entropy'] = 0.0
    df_features['protocol_entropy'] = 0.0
    df_features['unique_src_ips'] = 0
    df_features['unique_dst_ips'] = 0
    df_features['unique_src_ports'] = 0
    df_features['unique_dst_ports'] = 0
    df_features['top10_src_ip_fraction'] = 0.0
    df_features['top10_dst_ip_fraction'] = 0.0
    
    # Calculate entropy for each window
    for i in range(n_windows + 1):
        start_idx = i * window_size
        end_idx = min((i + 1) * window_size, len(df))
        
        if start_idx >= len(df):
            break
        
        window_df = df.iloc[start_idx:end_idx]
        
        # Count occurrences for entropy calculation (handle column name variations)
        src_ip_col = None
        dst_ip_col = None
        src_port_col = None
        dst_port_col = None
        protocol_col = None
        
        for col in window_df.columns:
            col_lower = col.strip().lower()
            if 'source ip' in col_lower or ('src' in col_lower and 'ip' in col_lower):
                src_ip_col = col
            elif 'destination ip' in col_lower or ('dst' in col_lower and 'ip' in col_lower) or ('dest' in col_lower and 'ip' in col_lower):
                dst_ip_col = col
            elif 'source port' in col_lower or ('src' in col_lower and 'port' in col_lower):
                src_port_col = col
            elif 'destination port' in col_lower or ('dst' in col_lower and 'port' in col_lower) or ('dest' in col_lower and 'port' in col_lower):
                dst_port_col = col
            elif 'protocol' in col_lower:
                protocol_col = col
        
        # Calculate counts (use available columns)
        src_ip_counts = window_df[src_ip_col].value_counts().to_dict() if src_ip_col else {}
        dst_ip_counts = window_df[dst_ip_col].value_counts().to_dict() if dst_ip_col else {}
        src_port_counts = window_df[src_port_col].value_counts().to_dict() if src_port_col else {}
        dst_port_counts = window_df[dst_port_col].value_counts().to_dict() if dst_port_col else {}
        protocol_counts = window_df[protocol_col].value_counts().to_dict() if protocol_col else {}
        
        # Calculate entropy for this window
        window_total = len(window_df)
        src_ip_entropy = calculate_entropy(src_ip_counts, window_total)
        dst_ip_entropy = calculate_entropy(dst_ip_counts, window_total)
        src_port_entropy = calculate_entropy(src_port_counts, window_total)
        dst_port_entropy = calculate_entropy(dst_port_counts, window_total)
        protocol_entropy = calculate_entropy(protocol_counts, window_total)
        
        # Calculate unique counts
        unique_src_ips = len(src_ip_counts)
        unique_dst_ips = len(dst_ip_counts)
        unique_src_ports = len(src_port_counts)
        unique_dst_ports = len(dst_port_counts)
        
        # Calculate Top-N fractions
        top10_src_fraction = calculate_top_n_fraction(src_ip_counts, window_total, 10)
        top10_dst_fraction = calculate_top_n_fraction(dst_ip_counts, window_total, 10)
        
        # Assign to all rows in this window
        df_features.loc[start_idx:end_idx-1, 'src_ip_entropy'] = src_ip_entropy
        df_features.loc[start_idx:end_idx-1, 'dst_ip_entropy'] = dst_ip_entropy
        df_features.loc[start_idx:end_idx-1, 'src_port_entropy'] = src_port_entropy
        df_features.loc[start_idx:end_idx-1, 'dst_port_entropy'] = dst_port_entropy
        df_features.loc[start_idx:end_idx-1, 'protocol_entropy'] = protocol_entropy
        df_features.loc[start_idx:end_idx-1, 'unique_src_ips'] = unique_src_ips
        df_features.loc[start_idx:end_idx-1, 'unique_dst_ips'] = unique_dst_ips
        df_features.loc[start_idx:end_idx-1, 'unique_src_ports'] = unique_src_ports
        df_features.loc[start_idx:end_idx-1, 'unique_dst_ports'] = unique_dst_ports
        df_features.loc[start_idx:end_idx-1, 'top10_src_ip_fraction'] = top10_src_fraction
        df_features.loc[start_idx:end_idx-1, 'top10_dst_ip_fraction'] = top10_dst_fraction
    
    # Packet size entropy (using coefficient of variation)
    df_features['packet_size_entropy'] = df_features['packet_size_std'] / (df_features['packet_size_mean'].replace(0, 1) + 1e-6)
    
    # Flow count (always 1 per row in CSV, but useful for aggregation)
    df_features['flow_count'] = 1
    
    # Fill NaN values
    df_features = df_features.fillna(0)
    
    print(f"Extracted {len(df_features.columns)} features")
    print(f"  Entropy features: src_ip={df_features['src_ip_entropy'].mean():.4f}, "
          f"dst_ip={df_features['dst_ip_entropy'].mean():.4f}, "
          f"src_port={df_features['src_port_entropy'].mean():.4f}")
    print(f"  Unique counts: src_ips={df_features['unique_src_ips'].mean():.2f}, "
          f"dst_ips={df_features['unique_dst_ips'].mean():.2f}")
    
    return df_features

def train_model(X, y, n_estimators=100, max_depth=10):
    """Train Random Forest model."""
    print(f"\nTraining Random Forest (n_estimators={n_estimators}, max_depth={max_depth})...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train
    model.fit(X_train, y_train)
    
    # Evaluate
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"5-fold CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Detailed metrics
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    # ROC curve
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC: {roc_auc:.4f}")
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Random Forest')
    plt.legend(loc="lower right")
    
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/ml_roc_curve.png")
    print("\nROC curve saved to results/ml_roc_curve.png")
    
    # Return all metrics needed for report
    return {
        'model': model,
        'X_test': X_test,
        'y_test': y_test,
        'X_train': X_train,
        'y_train': y_train,
        'train_score': train_score,
        'test_score': test_score,
        'cv_scores': cv_scores,
        'roc_auc': roc_auc
    }

def main():
    """Main training function."""
    print("=== Random Forest Model Training ===")
    
    # Load data
    df, csv_files = load_training_data()
    if df is None:
        print("Failed to load data. Exiting.")
        return
    
    # Extract features
    X = extract_features_from_csv(df)
    y = df['label'].values
    
    print(f"\nFeature extraction complete:")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Feature columns: {list(X.columns)[:10]}...")
    
    # Preprocessing (optional but recommended)
    print("\n=== Preprocessing ===")
    # Determine number of features for selection (at least 5, but not more than available)
    n_features_to_select = min(15, max(5, len(X.columns) - 5))
    preprocessor = FeaturePreprocessor(
        use_scaling=False,  # Random Forest doesn't need scaling, but can help
        use_outlier_removal=True,  # Remove outliers
        use_feature_selection=False,  # Disable for now (can enable after verifying features)
        n_features=n_features_to_select
    )
    
    # Ensure X is DataFrame with correct shape
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Verify X has correct number of rows
    if len(X) != len(y):
        print(f"ERROR: Shape mismatch before preprocessing! X: {X.shape}, y: {y.shape}")
        raise ValueError(f"Feature extraction returned wrong number of rows: {len(X)} vs {len(y)}")
    
    X_processed = preprocessor.fit_transform(X, y)
    print(f"Features after preprocessing: {X_processed.shape}")
    
    # Verify shapes match after preprocessing
    if len(X_processed) != len(y):
        print(f"WARNING: Preprocessing changed shape! X_processed: {X_processed.shape}, y: {y.shape}")
        print("Using original X without preprocessing...")
        X_processed = X
    
    # Save preprocessor
    os.makedirs("models", exist_ok=True)
    preprocessor.save("models/preprocessor.joblib")
    
    # Train model with improved hyperparameters
    training_results = train_model(X_processed, y, n_estimators=200, max_depth=15)
    model = training_results['model']
    X_test = training_results['X_test']
    y_test = training_results['y_test']
    train_score = training_results['train_score']
    test_score = training_results['test_score']
    cv_scores = training_results['cv_scores']
    roc_auc = training_results['roc_auc']
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/rf_model.joblib"
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Feature importance
    feature_names = X.columns.tolist()
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\nTop 10 Feature Importances:")
    for i in range(min(10, len(indices))):
        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Save training report
    save_training_report(model, X_test, y_test, feature_names, importances, indices, 
                        train_score, test_score, cv_scores, roc_auc, len(df), csv_files)

def save_training_report(model, X_test, y_test, feature_names, importances, indices,
                        train_score, test_score, cv_scores, roc_auc, total_rows, csv_files):
    """Save comprehensive training report to reports folder."""
    os.makedirs("reports", exist_ok=True)
    
    from sklearn.metrics import classification_report, confusion_matrix
    import json
    
    report_file = "reports/training_report.txt"
    json_file = "reports/training_metrics.json"
    
    # Generate predictions for report
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    
    # Write text report
    with open(report_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("DDoS Detection Model - Training Report\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total Training Samples: {total_rows:,}\n")
        f.write(f"CSV Files Used: {len(csv_files)}\n")
        f.write(f"Test Samples: {len(y_test):,}\n")
        f.write(f"Training Samples: {total_rows - len(y_test):,}\n\n")
        
        f.write("MODEL CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"Algorithm: Random Forest Classifier\n")
        f.write(f"Number of Estimators: {model.n_estimators}\n")
        f.write(f"Max Depth: {model.max_depth}\n")
        f.write(f"Number of Features: {len(feature_names)}\n\n")
        
        f.write("PERFORMANCE METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Train Accuracy: {train_score:.4f} ({train_score*100:.2f}%)\n")
        f.write(f"Test Accuracy: {test_score:.4f} ({test_score*100:.2f}%)\n")
        f.write(f"5-Fold CV F1 Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("DETAILED METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall (TPR): {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write(f"False Positive Rate (FPR): {fpr:.4f}\n\n")
        
        f.write("CONFUSION MATRIX\n")
        f.write("-" * 80 + "\n")
        f.write("                Predicted\n")
        f.write("              Benign  Attack\n")
        f.write(f"Actual Benign   {tn:5d}   {fp:5d}\n")
        f.write(f"       Attack   {fn:5d}   {tp:5d}\n\n")
        
        f.write("CLASSIFICATION REPORT\n")
        f.write("-" * 80 + "\n")
        f.write(classification_report(y_test, y_pred, target_names=['Benign', 'Attack']))
        f.write("\n")
        
        f.write("FEATURE IMPORTANCE (Top 10)\n")
        f.write("-" * 80 + "\n")
        for i in range(min(10, len(indices))):
            f.write(f"{i+1:2d}. {feature_names[indices[i]]:30s}: {importances[indices[i]]:.4f}\n")
        f.write("\n")
        
        f.write("FILES GENERATED\n")
        f.write("-" * 80 + "\n")
        f.write(f"Model: models/rf_model.joblib\n")
        f.write(f"Preprocessor: models/preprocessor.joblib\n")
        f.write(f"ROC Curve: results/ml_roc_curve.png\n")
        f.write(f"Training Report: reports/training_report.txt\n")
        f.write(f"Metrics JSON: reports/training_metrics.json\n")
    
    # Save JSON metrics for dashboard
    metrics_json = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'algorithm': 'Random Forest',
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'n_features': len(feature_names)
        },
        'dataset': {
            'total_samples': int(total_rows),
            'csv_files_count': len(csv_files),
            'test_samples': int(len(y_test)),
            'train_samples': int(total_rows - len(y_test))
        },
        'metrics': {
            'train_accuracy': float(train_score),
            'test_accuracy': float(test_score),
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'roc_auc': float(roc_auc),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'false_positive_rate': float(fpr)
        },
        'confusion_matrix': {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp)
        },
        'top_features': [
            {'name': feature_names[indices[i]], 'importance': float(importances[indices[i]])}
            for i in range(min(10, len(indices)))
        ],
        'files': {
            'model': 'models/rf_model.joblib',
            'preprocessor': 'models/preprocessor.joblib',
            'roc_curve': 'results/ml_roc_curve.png',
            'report': 'reports/training_report.txt'
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(metrics_json, f, indent=2)
    
    print(f"\nTraining report saved to {report_file}")
    print(f"Metrics JSON saved to {json_file}")

if __name__ == "__main__":
    main()

