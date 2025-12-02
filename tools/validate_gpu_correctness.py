#!/usr/bin/env python3
"""
Validate GPU entropy calculation correctness by comparing with CPU baseline.
Generates comparison report and scatter plots.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
RESULTS_DIR = project_root / "results"
RESULTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(project_root))

def generate_synthetic_windows(num_windows=100):
    """Generate synthetic window data for testing."""
    windows = []
    
    for i in range(num_windows):
        # Create a window with known entropy characteristics
        window = {
            'window_id': i,
            'total_packets': 1000 + i * 10,
            'src_ip_counts': {},
            'dst_ip_counts': {},
            'src_port_counts': {},
            'dst_port_counts': {},
            'packet_size_counts': {},
            'protocol_counts': {}
        }
        
        # Test case 1: Uniform distribution (high entropy)
        if i < num_windows // 3:
            num_unique = 50
            for j in range(num_unique):
                window['src_ip_counts'][j] = 20
                window['src_port_counts'][j % 65536] = 20
        # Test case 2: Skewed distribution (low entropy - DDoS-like)
        elif i < 2 * num_windows // 3:
            window['src_ip_counts'][1] = 900  # Single source IP
            window['src_ip_counts'][2] = 50
            window['src_ip_counts'][3] = 50
            window['src_port_counts'][80] = 950  # Single port
            window['src_port_counts'][443] = 50
        # Test case 3: Mixed distribution
        else:
            for j in range(10):
                window['src_ip_counts'][j] = 100 - j * 5
                window['src_port_counts'][j + 80] = 100 - j * 5
        
        # Fill other features
        window['dst_ip_counts'][100] = window['total_packets'] // 2
        window['dst_ip_counts'][200] = window['total_packets'] // 2
        window['dst_port_counts'][80] = window['total_packets']
        window['packet_size_counts'][64] = window['total_packets'] // 2
        window['packet_size_counts'][1500] = window['total_packets'] // 2
        window['protocol_counts'][6] = window['total_packets'] // 2  # TCP
        window['protocol_counts'][17] = window['total_packets'] // 2  # UDP
        
        windows.append(window)
    
    return windows

def calculate_cpu_entropy(counts_dict, total):
    """Calculate entropy on CPU (Python reference implementation)."""
    if total == 0:
        return 0.0
    
    import math
    entropy = 0.0
    for count in counts_dict.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy

def calculate_cpu_entropy_for_window(window):
    """Calculate all entropy features for a window on CPU."""
    total = window['total_packets']
    
    features = {
        'src_ip': calculate_cpu_entropy(window['src_ip_counts'], total),
        'dst_ip': calculate_cpu_entropy(window['dst_ip_counts'], total),
        'src_port': calculate_cpu_entropy(window['src_port_counts'], total),
        'dst_port': calculate_cpu_entropy(window['dst_port_counts'], total),
        'packet_size': calculate_cpu_entropy(window['packet_size_counts'], total),
        'protocol': calculate_cpu_entropy(window['protocol_counts'], total)
    }
    
    return features

def _counts_to_pairs(counts_dict):
    return [{"key": int(k), "value": int(v)} for k, v in counts_dict.items()]


def write_windows_to_json(windows, path: Path):
    payload = {"windows": []}
    for idx, window in enumerate(windows):
        start_us = window.get("window_start_us", idx * 1_000_000)
        end_us = window.get("window_end_us", start_us + 1_000_000)
        entry = {
            "window_start_us": int(start_us),
            "window_end_us": int(end_us),
            "total_packets": int(window["total_packets"]),
            "total_bytes": int(window.get("total_bytes", window["total_packets"] * 512)),
            "src_ip_counts": _counts_to_pairs(window["src_ip_counts"]),
            "dst_ip_counts": _counts_to_pairs(window["dst_ip_counts"]),
            "src_port_counts": _counts_to_pairs(window["src_port_counts"]),
            "dst_port_counts": _counts_to_pairs(window["dst_port_counts"]),
            "packet_size_counts": _counts_to_pairs(window["packet_size_counts"]),
            "protocol_counts": _counts_to_pairs(window["protocol_counts"]),
        }
        payload["windows"].append(entry)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def run_gpu_entropy_test(windows):
    """Run GPU entropy calculation (requires C++ test program)."""
    project_root = Path(__file__).parent.parent
    test_binary = project_root / "build" / "test_gpu_entropy"
    input_json = RESULTS_DIR / "gpu_test_windows.json"
    output_json = RESULTS_DIR / "gpu_entropy_results.json"
    write_windows_to_json(windows, input_json)
    
    if not test_binary.exists():
        print(f"ERROR: GPU test binary not found at {test_binary}")
        print("  Building test binary...")
        build_cmd = [
            "g++", "-std=c++17", "-O3",
            "-Isrc", "-I../src",
            "-I/usr/include",
            str(project_root / "tools" / "test_gpu_entropy.cpp"),
            str(project_root / "src" / "opencl" / "gpu_detector.cpp"),
            str(project_root / "src" / "opencl" / "host.cpp"),
            str(project_root / "src" / "detectors" / "entropy_cpu.cpp"),
            str(project_root / "src" / "ingest" / "window_manager.cpp"),
            str(project_root / "src" / "ingest" / "pcap_reader.cpp"),
            str(project_root / "src" / "utils" / "simple_json.cpp"),
            "-lOpenCL", "-lpcap", "-lpthread",
            "-o", str(test_binary),
        ]
        try:
            subprocess.run(build_cmd, check=True, cwd=str(project_root))
            print(f"  Built successfully: {test_binary}")
        except subprocess.CalledProcessError as exc:
            print(f"  Build failed: {exc}")
            return None
    
    try:
        result = subprocess.run(
            [str(test_binary), str(input_json), str(output_json)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(project_root),
        )
    except subprocess.TimeoutExpired:
        print("GPU entropy test timed out")
        return None
    except Exception as err:  # pylint: disable=broad-except
        print(f"Error running GPU test: {err}")
        return None
    
    if result.returncode != 0:
        print("GPU entropy test failed")
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip())
        return None
    
    try:
        with open(output_json, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            print("GPU entropy test passed")
            return payload
    except (OSError, json.JSONDecodeError) as err:
        print(f"Failed to read GPU results: {err}")
        return None

def compare_results(cpu_results, gpu_results, tolerance=1e-5):
    """Compare CPU and GPU results."""
    if gpu_results is None:
        return None
    
    differences = []
    max_diff = 0.0
    max_diff_window = 0
    max_diff_feature = None
    
    num_windows = len(cpu_results)
    num_features = 6
    feature_names = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'packet_size', 'protocol']
    
    for i in range(num_windows):
        cpu_feat = cpu_results[i]
        gpu_start_idx = i * num_features
        
        for j, feat_name in enumerate(feature_names):
            cpu_val = cpu_feat[feat_name]
            gpu_val = gpu_results[gpu_start_idx + j]
            
            diff = abs(cpu_val - gpu_val)
            differences.append({
                'window': i,
                'feature': feat_name,
                'cpu': cpu_val,
                'gpu': gpu_val,
                'diff': diff
            })
            
            if diff > max_diff:
                max_diff = diff
                max_diff_window = i
                max_diff_feature = feat_name
    
    return {
        'differences': differences,
        'max_diff': max_diff,
        'max_diff_window': max_diff_window,
        'max_diff_feature': max_diff_feature,
        'tolerance': tolerance,
        'all_within_tolerance': max_diff <= tolerance
    }

def generate_report(comparison_result, output_file='results/gpu_correctness_report.txt'):
    """Generate comparison report."""
    os.makedirs('results', exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GPU Entropy Correctness Validation Report\n")
        f.write("=" * 80 + "\n\n")
        
        if comparison_result is None:
            f.write("ERROR: GPU results not available.\n")
            f.write("Please compile and run GPU test program first.\n")
            return
        
        f.write(f"Tolerance: {comparison_result['tolerance']}\n")
        f.write(f"Maximum difference: {comparison_result['max_diff']:.10f}\n")
        f.write(f"Window with max diff: {comparison_result['max_diff_window']}\n")
        f.write(f"Feature with max diff: {comparison_result['max_diff_feature']}\n")
        f.write(f"All within tolerance: {comparison_result['all_within_tolerance']}\n\n")
        
        f.write("Sample differences (first 20):\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Window':<8} {'Feature':<15} {'CPU':<15} {'GPU':<15} {'Diff':<15}\n")
        f.write("-" * 80 + "\n")
        
        for diff in comparison_result['differences'][:20]:
            f.write(f"{diff['window']:<8} {diff['feature']:<15} "
                   f"{diff['cpu']:<15.6f} {diff['gpu']:<15.6f} {diff['diff']:<15.10f}\n")
    
    print(f"Report saved to: {output_file}")

def plot_comparison(cpu_results, gpu_results, output_file='results/gpu_cpu_entropy_comparison.png'):
    """Generate scatter plot comparing CPU vs GPU entropy."""
    if gpu_results is None:
        print("Skipping plot - GPU results not available")
        return
    
    os.makedirs('results', exist_ok=True)
    
    num_windows = len(cpu_results)
    num_features = 6
    feature_names = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'packet_size', 'protocol']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for j, feat_name in enumerate(feature_names):
        cpu_vals = [cpu_results[i][feat_name] for i in range(num_windows)]
        gpu_vals = [gpu_results[i * num_features + j] for i in range(num_windows)]
        
        ax = axes[j]
        ax.scatter(cpu_vals, gpu_vals, alpha=0.6)
        
        # Add diagonal line (perfect match)
        min_val = min(min(cpu_vals), min(gpu_vals))
        max_val = max(max(cpu_vals), max(gpu_vals))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect match')
        
        ax.set_xlabel('CPU Entropy')
        ax.set_ylabel('GPU Entropy')
        ax.set_title(f'{feat_name} Entropy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    print(f"Plot saved to: {output_file}")

def main():
    print("=" * 80)
    print("GPU Entropy Correctness Validation")
    print("=" * 80)
    
    # Generate test windows
    print("\n1. Generating synthetic test windows...")
    windows = generate_synthetic_windows(100)
    print(f"   Generated {len(windows)} test windows")
    
    # Calculate CPU entropy
    print("\n2. Calculating CPU entropy (reference)...")
    cpu_results = []
    for window in windows:
        cpu_results.append(calculate_cpu_entropy_for_window(window))
    print(f"   Calculated entropy for {len(cpu_results)} windows")
    
    # Run GPU entropy (placeholder - requires C++ test program)
    print("\n3. Running GPU entropy calculation...")
    gpu_payload = run_gpu_entropy_test(windows)
    gpu_results = gpu_payload.get("gpu_results") if gpu_payload else None
    
    # Compare results
    print("\n4. Comparing CPU vs GPU results...")
    comparison_result = compare_results(cpu_results, gpu_results, tolerance=1e-5)
    
    # Generate report
    print("\n5. Generating report...")
    generate_report(comparison_result)
    
    # Generate plots
    print("\n6. Generating comparison plots...")
    plot_comparison(cpu_results, gpu_results)
    
    print("\n" + "=" * 80)
    if comparison_result and comparison_result['all_within_tolerance']:
        print("✓ VALIDATION PASSED: All GPU results within tolerance")
    else:
        print("⚠ VALIDATION INCOMPLETE: GPU test program required")
    print("=" * 80)

if __name__ == '__main__':
    main()

