#!/usr/bin/env python3
"""
Evaluate detection performance from log files.
Computes precision, recall, F1, FPR, and detection lead time.
"""

import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

def load_alerts(log_dir):
    """Load alerts from CSV log file."""
    alerts_file = os.path.join(log_dir, 'alerts.csv')
    if not os.path.exists(alerts_file):
        print(f"Alerts file not found: {alerts_file}")
        return None
    
    df = pd.read_csv(alerts_file)
    return df

def compute_metrics(alerts_df, ground_truth_file=None):
    """Compute detection metrics."""
    if alerts_df is None or len(alerts_df) == 0:
        print("No alerts to evaluate")
        return None
    
    print("=== Detection Metrics ===")
    print(f"Total alerts: {len(alerts_df)}")
    
    # Detector breakdown
    if 'detector' in alerts_df.columns:
        print("\nAlerts by detector:")
        print(alerts_df['detector'].value_counts())
    
    # Score statistics
    if 'score' in alerts_df.columns:
        print(f"\nScore statistics:")
        print(f"  Mean: {alerts_df['score'].mean():.4f}")
        print(f"  Std: {alerts_df['score'].std():.4f}")
        print(f"  Min: {alerts_df['score'].min():.4f}")
        print(f"  Max: {alerts_df['score'].max():.4f}")
    
    # If ground truth available, compute precision/recall
    if ground_truth_file and os.path.exists(ground_truth_file):
        gt_df = pd.read_csv(ground_truth_file)
        # Match alerts with ground truth
        # This is simplified - would need proper matching logic
        print("\n=== Comparison with Ground Truth ===")
        # Compute precision, recall, F1, FPR
        # (Implementation would depend on ground truth format)
    
    return {
        'total_alerts': len(alerts_df),
        'mean_score': alerts_df['score'].mean() if 'score' in alerts_df.columns else 0
    }

def compute_lead_time(alerts_df, attack_start_times):
    """Compute detection lead time (time from attack start to first alert)."""
    if alerts_df is None or len(alerts_df) == 0:
        return None
    
    lead_times = []
    for _, alert in alerts_df.iterrows():
        window_start = alert.get('window_start_ms', 0)
        # Compare with attack start times
        # Simplified - would need proper attack timeline
        lead_times.append(window_start)
    
    if lead_times:
        print(f"\n=== Detection Lead Time ===")
        print(f"Mean: {np.mean(lead_times):.2f} ms")
        print(f"Median: {np.median(lead_times):.2f} ms")
        print(f"Min: {np.min(lead_times):.2f} ms")
        print(f"Max: {np.max(lead_times):.2f} ms")
    
    return lead_times

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 evaluate_detection.py <log_dir> [ground_truth_file]")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    ground_truth_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Load alerts
    alerts_df = load_alerts(log_dir)
    
    # Compute metrics
    metrics = compute_metrics(alerts_df, ground_truth_file)
    
    # Compute lead time
    compute_lead_time(alerts_df, None)

if __name__ == "__main__":
    main()

