#!/usr/bin/env python3
"""
Performance benchmarks for Phase 2 CPU baseline implementation.
Measures processing speed, memory usage, and scalability.
"""

import sys
import os
import time
import subprocess
import psutil
from pathlib import Path
import json

def benchmark_entropy_calculation():
    """Benchmark entropy calculation performance."""
    print("=" * 80)
    print("Benchmark: Entropy Calculation")
    print("=" * 80)
    
    # Simulate different histogram sizes
    test_cases = [
        ("Small (10 unique values)", 10),
        ("Medium (100 unique values)", 100),
        ("Large (1000 unique values)", 1000),
        ("Very Large (10000 unique values)", 10000),
    ]
    
    results = []
    
    for name, n_unique in test_cases:
        # Create test histogram
        histogram = {i: 100 for i in range(n_unique)}
        total = sum(histogram.values())
        
        # Benchmark
        iterations = 10000
        start_time = time.time()
        
        for _ in range(iterations):
            entropy = 0.0
            import math
            for count in histogram.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
        
        elapsed = time.time() - start_time
        time_per_calc = (elapsed / iterations) * 1000  # milliseconds
        
        results.append({
            'name': name,
            'n_unique': n_unique,
            'time_per_calc_ms': time_per_calc,
            'throughput_calc_per_sec': 1000 / time_per_calc if time_per_calc > 0 else 0
        })
        
        print(f"\n{name}:")
        print(f"  Time per calculation: {time_per_calc:.4f} ms")
        print(f"  Throughput: {1000/time_per_calc:.0f} calculations/sec")
    
    return results

def benchmark_window_processing():
    """Benchmark window processing performance."""
    print("\n" + "=" * 80)
    print("Benchmark: Window Processing")
    print("=" * 80)
    
    # Simulate packet processing
    packets_per_second = [1000, 10000, 100000, 500000]
    window_size_sec = 1
    
    results = []
    
    for pps in packets_per_second:
        n_packets = pps * window_size_sec
        
        # Simulate window processing
        start_time = time.time()
        
        # Simulate histogram updates
        histogram = {}
        for i in range(n_packets):
            key = i % 100  # 100 unique values
            histogram[key] = histogram.get(key, 0) + 1
        
        elapsed = time.time() - start_time
        
        results.append({
            'packets_per_sec': pps,
            'n_packets': n_packets,
            'processing_time_ms': elapsed * 1000,
            'throughput_pps': n_packets / elapsed if elapsed > 0 else 0
        })
        
        print(f"\n{pps:,} packets/sec:")
        print(f"  Processing time: {elapsed*1000:.2f} ms")
        print(f"  Throughput: {n_packets/elapsed:,.0f} packets/sec")
    
    return results

def benchmark_memory_usage():
    """Benchmark memory usage for different window sizes."""
    print("\n" + "=" * 80)
    print("Benchmark: Memory Usage")
    print("=" * 80)
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
    
    test_cases = [
        ("Small window (100 packets)", 100),
        ("Medium window (1000 packets)", 1000),
        ("Large window (10000 packets)", 10000),
        ("Very Large window (100000 packets)", 100000),
    ]
    
    results = []
    
    for name, n_packets in test_cases:
        # Simulate window with histograms
        histograms = {
            'src_ip': {},
            'dst_ip': {},
            'src_port': {},
            'dst_port': {},
            'packet_size': {},
            'protocol': {}
        }
        
        for i in range(n_packets):
            for hist_name in histograms:
                key = i % 1000  # 1000 unique values per histogram
                histograms[hist_name][key] = histograms[hist_name].get(key, 0) + 1
        
        memory_used = (process.memory_info().rss / (1024 * 1024)) - initial_memory
        
        results.append({
            'name': name,
            'n_packets': n_packets,
            'memory_mb': memory_used
        })
        
        print(f"\n{name}:")
        print(f"  Memory usage: {memory_used:.2f} MB")
        print(f"  Memory per packet: {memory_used*1024*1024/n_packets:.2f} bytes")
        
        # Cleanup
        del histograms
    
    return results

def save_benchmark_results(results):
    """Save benchmark results to JSON file."""
    os.makedirs("results", exist_ok=True)
    output_file = "results/phase2_benchmarks.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark results saved to {output_file}")

def main():
    """Main benchmark function."""
    print("=" * 80)
    print("Phase 2: Performance Benchmarks")
    print("=" * 80)
    
    results = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'entropy_calculation': benchmark_entropy_calculation(),
        'window_processing': benchmark_window_processing(),
        'memory_usage': benchmark_memory_usage()
    }
    
    save_benchmark_results(results)
    
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    
    # Summary
    print("\nSummary:")
    print(f"  Entropy calculation: {results['entropy_calculation'][-1]['throughput_calc_per_sec']:.0f} calc/sec")
    print(f"  Window processing: {results['window_processing'][-1]['throughput_pps']:,.0f} packets/sec")
    print(f"  Memory efficiency: {results['memory_usage'][-1]['memory_per_packet']:.2f} bytes/packet")

if __name__ == "__main__":
    main()

