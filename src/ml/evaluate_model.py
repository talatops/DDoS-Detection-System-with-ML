#!/usr/bin/env python3
"""
Evaluate trained ML model on test data.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from src.ml.feature_extractor import FeatureExtractor

def evaluate_model(model_path="models/rf_model.joblib", test_data_path=None):
    """Evaluate model on test data."""
    print("=== Model Evaluation ===")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    print("Model loaded successfully")
    
    # Load test data
    if test_data_path:
        print(f"Loading test data from {test_data_path}...")
        df_test = pd.read_csv(test_data_path)
        
        # Extract features (same as training)
        X_test = extract_features_from_csv(df_test)
        y_test = df_test['label'].values if 'label' in df_test.columns else None
        
        if y_test is not None:
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Metrics
            print("\n=== Evaluation Metrics ===")
            print(classification_report(y_test, y_pred))
            print("\nConfusion Matrix:")
            print(confusion_matrix(y_test, y_pred))
            
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            print(f"\nROC AUC: {roc_auc:.4f}")
            
            # Plot ROC curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig("results/roc_curve_eval.png")
            print("ROC curve saved to results/roc_curve_eval.png")
            
            # Precision-Recall curve
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.savefig("results/pr_curve_eval.png")
            print("PR curve saved to results/pr_curve_eval.png")
    
    # Feature importance
    print("\n=== Feature Importance ===")
    feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("Top 10 Features:")
    for i in range(min(10, len(indices))):
        print(f"  {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")

def extract_features_from_csv(df):
    """Extract features (same as training)."""
    df_features = pd.DataFrame()
    df_features['total_packets'] = df.get('Total Fwd Packets', 0) + df.get('Total Backward Packets', 0)
    df_features['total_bytes'] = df.get('Total Length of Fwd Packets', 0) + df.get('Total Length of Bwd Packets', 0)
    df_features['unique_src_ips'] = 1
    df_features['unique_dst_ips'] = 1
    df_features['unique_src_ports'] = 1
    df_features['unique_dst_ports'] = 1
    df_features['flow_count'] = 1
    df_features['avg_packet_size'] = df_features['total_bytes'] / df_features['total_packets'].replace(0, 1)
    df_features['src_ip_entropy'] = 0.0
    df_features['dst_ip_entropy'] = 0.0
    df_features['src_port_entropy'] = 0.0
    df_features['dst_port_entropy'] = 0.0
    df_features['packet_size_entropy'] = df.get('Packet Length Std', 0) / (df.get('Packet Length Mean', 1) + 1e-6)
    df_features['protocol_entropy'] = 0.0
    df_features['top10_src_ip_fraction'] = 1.0
    df_features['top10_dst_ip_fraction'] = 1.0
    return df_features.fillna(0)

if __name__ == "__main__":
    evaluate_model()

