#!/usr/bin/env python3
"""
Generate plots from experiment results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

def plot_metrics(log_dir):
    """Plot system metrics over time."""
    metrics_file = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        print(f"Metrics file not found: {metrics_file}")
        return
    
    df = pd.read_csv(metrics_file)
    if len(df) == 0:
        print("No metrics data")
        return
    
    # Convert timestamp to datetime
    df['time'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # GPU utilization
    axes[0, 0].plot(df['time'], df['gpu_percent'])
    axes[0, 0].set_title('GPU Utilization')
    axes[0, 0].set_ylabel('Utilization (%)')
    axes[0, 0].grid(True)
    
    # Throughput
    axes[0, 1].plot(df['time'], df['pps_in'])
    axes[0, 1].set_title('Throughput')
    axes[0, 1].set_ylabel('Packets per Second')
    axes[0, 1].grid(True)
    
    # CPU utilization
    axes[1, 0].plot(df['time'], df['cpu_percent'])
    axes[1, 0].set_title('CPU Utilization')
    axes[1, 0].set_ylabel('Utilization (%)')
    axes[1, 0].grid(True)
    
    # Memory usage
    axes[1, 1].plot(df['time'], df['memory_mb'])
    axes[1, 1].set_title('Memory Usage')
    axes[1, 1].set_ylabel('Memory (MB)')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, 'metrics_plot.png'))
    print(f"Metrics plot saved to {log_dir}/metrics_plot.png")

def plot_alerts(log_dir):
    """Plot alerts timeline."""
    alerts_file = os.path.join(log_dir, 'alerts.csv')
    if not os.path.exists(alerts_file):
        return
    
    df = pd.read_csv(alerts_file)
    if len(df) == 0:
        return
    
    df['time'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    
    plt.figure(figsize=(12, 6))
    plt.scatter(df['time'], df['score'], alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel('Detection Score')
    plt.title('Detection Alerts Timeline')
    plt.grid(True)
    plt.savefig(os.path.join(log_dir, 'alerts_plot.png'))
    print(f"Alerts plot saved to {log_dir}/alerts_plot.png")

def plot_kernel_times(log_dir):
    """Plot kernel execution times."""
    kernel_file = os.path.join(log_dir, 'kernel_times.csv')
    if not os.path.exists(kernel_file):
        return
    
    df = pd.read_csv(kernel_file)
    if len(df) == 0:
        return
    
    plt.figure(figsize=(12, 6))
    df.boxplot(column='execution_time_ms', by='kernel_name')
    plt.xlabel('Kernel')
    plt.ylabel('Execution Time (ms)')
    plt.title('Kernel Execution Times')
    plt.savefig(os.path.join(log_dir, 'kernel_times_plot.png'))
    print(f"Kernel times plot saved to {log_dir}/kernel_times_plot.png")

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <log_dir>")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    
    print(f"Generating plots from {log_dir}...")
    plot_metrics(log_dir)
    plot_alerts(log_dir)
    plot_kernel_times(log_dir)
    print("Done!")

if __name__ == "__main__":
    main()

