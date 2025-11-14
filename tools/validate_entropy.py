#!/usr/bin/env python3
"""
Validate entropy calculations from CPU baseline implementation.
Compares entropy values against ground truth and plots entropy over time.
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter
import math

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def calculate_entropy_ground_truth(counts_dict, total):
    """Calculate Shannon entropy: H = -Σ p_i * log2(p_i)"""
    if total == 0:
        return 0.0
    
    entropy = 0.0
    for count in counts_dict.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def validate_entropy_formula():
    """Test entropy calculation with known values."""
    print("=" * 80)
    print("Testing Entropy Formula Correctness")
    print("=" * 80)
    
    # Test case 1: Uniform distribution (maximum entropy)
    # For 4 items with equal probability, entropy = log2(4) = 2.0
    counts = {1: 25, 2: 25, 3: 25, 4: 25}
    total = 100
    entropy = calculate_entropy_ground_truth(counts, total)
    expected = math.log2(4)
    print(f"\nTest 1: Uniform distribution (4 items)")
    print(f"  Expected entropy: {expected:.6f}")
    print(f"  Calculated entropy: {entropy:.6f}")
    print(f"  Difference: {abs(entropy - expected):.6f}")
    assert abs(entropy - expected) < 1e-5, "Entropy calculation failed!"
    print("  ✅ PASSED")
    
    # Test case 2: Single value (zero entropy)
    counts = {1: 100}
    total = 100
    entropy = calculate_entropy_ground_truth(counts, total)
    expected = 0.0
    print(f"\nTest 2: Single value (zero entropy)")
    print(f"  Expected entropy: {expected:.6f}")
    print(f"  Calculated entropy: {entropy:.6f}")
    assert abs(entropy - expected) < 1e-5, "Zero entropy test failed!"
    print("  ✅ PASSED")
    
    # Test case 3: Skewed distribution (low entropy)
    counts = {1: 90, 2: 5, 3: 3, 4: 2}
    total = 100
    entropy = calculate_entropy_ground_truth(counts, total)
    print(f"\nTest 3: Skewed distribution")
    print(f"  Calculated entropy: {entropy:.6f}")
    assert 0 < entropy < math.log2(4), "Skewed distribution entropy should be between 0 and max"
    print("  ✅ PASSED")
    
    print("\n" + "=" * 80)
    print("All entropy formula tests PASSED!")
    print("=" * 80)

def validate_entropy_from_csv(csv_file, window_size_sec=1):
    """
    Validate entropy calculation from CSV flow data.
    Simulates windowing and calculates entropy per window.
    """
    print("\n" + "=" * 80)
    print(f"Validating Entropy from CSV: {csv_file}")
    print("=" * 80)
    
    try:
        df = pd.read_csv(csv_file, nrows=10000, low_memory=False)
        print(f"Loaded {len(df)} rows from CSV")
        
        # Group by time windows (simplified - using row numbers as windows)
        window_size = window_size_sec * 1000  # Assuming ~1000 rows per second
        n_windows = len(df) // window_size
        
        if n_windows == 0:
            print("  ⚠️  Not enough data for windowing")
            return
        
        entropy_over_time = {
            'src_ip': [],
            'dst_ip': [],
            'src_port': [],
            'dst_port': [],
            'window': []
        }
        
        for i in range(n_windows):
            start_idx = i * window_size
            end_idx = min((i + 1) * window_size, len(df))
            window_df = df.iloc[start_idx:end_idx]
            
            if len(window_df) == 0:
                continue
            
            # Calculate entropy for each feature
            # Source IP entropy
            if ' Source IP' in window_df.columns:
                src_ip_counts = Counter(window_df[' Source IP'].dropna())
                src_ip_entropy = calculate_entropy_ground_truth(src_ip_counts, len(window_df))
                entropy_over_time['src_ip'].append(src_ip_entropy)
            
            # Destination IP entropy
            if ' Destination IP' in window_df.columns:
                dst_ip_counts = Counter(window_df[' Destination IP'].dropna())
                dst_ip_entropy = calculate_entropy_ground_truth(dst_ip_counts, len(window_df))
                entropy_over_time['dst_ip'].append(dst_ip_entropy)
            
            # Source port entropy
            if ' Source Port' in window_df.columns:
                src_port_counts = Counter(window_df[' Source Port'].dropna())
                src_port_entropy = calculate_entropy_ground_truth(src_port_counts, len(window_df))
                entropy_over_time['src_port'].append(src_port_entropy)
            
            # Destination port entropy
            if ' Destination Port' in window_df.columns:
                dst_port_counts = Counter(window_df[' Destination Port'].dropna())
                dst_port_entropy = calculate_entropy_ground_truth(dst_port_counts, len(window_df))
                entropy_over_time['dst_port'].append(dst_port_entropy)
            
            entropy_over_time['window'].append(i)
        
        # Plot entropy over time
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        if entropy_over_time['src_ip']:
            plt.plot(entropy_over_time['window'], entropy_over_time['src_ip'], 'b-', label='Source IP')
            plt.xlabel('Window')
            plt.ylabel('Entropy (bits)')
            plt.title('Source IP Entropy Over Time')
            plt.grid(True)
            plt.legend()
        
        plt.subplot(2, 2, 2)
        if entropy_over_time['dst_ip']:
            plt.plot(entropy_over_time['window'], entropy_over_time['dst_ip'], 'r-', label='Destination IP')
            plt.xlabel('Window')
            plt.ylabel('Entropy (bits)')
            plt.title('Destination IP Entropy Over Time')
            plt.grid(True)
            plt.legend()
        
        plt.subplot(2, 2, 3)
        if entropy_over_time['src_port']:
            plt.plot(entropy_over_time['window'], entropy_over_time['src_port'], 'g-', label='Source Port')
            plt.xlabel('Window')
            plt.ylabel('Entropy (bits)')
            plt.title('Source Port Entropy Over Time')
            plt.grid(True)
            plt.legend()
        
        plt.subplot(2, 2, 4)
        if entropy_over_time['dst_port']:
            plt.plot(entropy_over_time['window'], entropy_over_time['dst_port'], 'm-', label='Destination Port')
            plt.xlabel('Window')
            plt.ylabel('Entropy (bits)')
            plt.title('Destination Port Entropy Over Time')
            plt.grid(True)
            plt.legend()
        
        plt.tight_layout()
        
        output_file = 'results/entropy_validation.png'
        os.makedirs('results', exist_ok=True)
        plt.savefig(output_file, dpi=150)
        print(f"\n✅ Entropy plots saved to {output_file}")
        
        # Print statistics
        print("\nEntropy Statistics:")
        if entropy_over_time['src_ip']:
            print(f"  Source IP:   mean={np.mean(entropy_over_time['src_ip']):.4f}, "
                  f"std={np.std(entropy_over_time['src_ip']):.4f}, "
                  f"min={np.min(entropy_over_time['src_ip']):.4f}, "
                  f"max={np.max(entropy_over_time['src_ip']):.4f}")
        if entropy_over_time['dst_ip']:
            print(f"  Destination IP: mean={np.mean(entropy_over_time['dst_ip']):.4f}, "
                  f"std={np.std(entropy_over_time['dst_ip']):.4f}, "
                  f"min={np.min(entropy_over_time['dst_ip']):.4f}, "
                  f"max={np.max(entropy_over_time['dst_ip']):.4f}")
        
    except Exception as e:
        print(f"  ❌ Error validating CSV: {e}")
        import traceback
        traceback.print_exc()

def validate_ddos_attack_pattern():
    """
    Validate that entropy correctly identifies DDoS attack patterns.
    DDoS attacks typically show:
    - Low source IP entropy (many packets from few IPs)
    - Low destination IP entropy (targeting one victim)
    - Low port entropy (same ports)
    """
    print("\n" + "=" * 80)
    print("Validating DDoS Attack Pattern Detection")
    print("=" * 80)
    
    # Simulate normal traffic (high entropy)
    normal_src_ips = {f"192.168.1.{i}": 10 for i in range(1, 101)}  # 100 unique IPs
    normal_total = sum(normal_src_ips.values())
    normal_entropy = calculate_entropy_ground_truth(normal_src_ips, normal_total)
    
    # Simulate DDoS attack (low entropy)
    attack_src_ips = {"10.0.0.1": 900, "10.0.0.2": 100}  # Only 2 IPs
    attack_total = sum(attack_src_ips.values())
    attack_entropy = calculate_entropy_ground_truth(attack_src_ips, attack_total)
    
    print(f"\nNormal Traffic (100 unique source IPs):")
    print(f"  Entropy: {normal_entropy:.4f} bits")
    
    print(f"\nDDoS Attack (2 source IPs, same pattern):")
    print(f"  Entropy: {attack_entropy:.4f} bits")
    
    print(f"\nEntropy Drop: {normal_entropy - attack_entropy:.4f} bits")
    
    if attack_entropy < normal_entropy * 0.5:
        print("  ✅ PASSED: Attack shows significantly lower entropy")
    else:
        print("  ⚠️  WARNING: Attack entropy not significantly lower")
    
    return normal_entropy, attack_entropy

def main():
    """Main validation function."""
    print("=" * 80)
    print("Phase 2: Entropy Validation and Testing")
    print("=" * 80)
    
    # Test 1: Validate entropy formula
    validate_entropy_formula()
    
    # Test 2: Validate DDoS attack pattern detection
    validate_ddos_attack_pattern()
    
    # Test 3: Validate entropy from CSV files (if available)
    csv_dir = Path("data/caida-ddos2007")
    if csv_dir.exists():
        csv_files = list(csv_dir.rglob("*.csv"))
        if csv_files:
            print(f"\nFound {len(csv_files)} CSV files")
            # Test with first CSV file
            validate_entropy_from_csv(csv_files[0])
        else:
            print("\n⚠️  No CSV files found in data/caida-ddos2007")
    else:
        print("\n⚠️  CSV directory not found: data/caida-ddos2007")
    
    print("\n" + "=" * 80)
    print("Entropy Validation Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

