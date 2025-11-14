#!/usr/bin/env python3
"""
Feature extraction for ML training and inference.
Extracts features from window statistics for Random Forest model.
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import json

class FeatureExtractor:
    """Extract features from window statistics for ML model."""
    
    def __init__(self, features_spec_file="src/ml/features_spec.json"):
        """Initialize feature extractor with feature specification."""
        self.features_spec = self._load_features_spec(features_spec_file)
    
    def _load_features_spec(self, spec_file: str) -> Dict:
        """Load feature specification from JSON file."""
        try:
            with open(spec_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default feature specification
            return {
                "features": [
                    "total_packets",
                    "total_bytes",
                    "unique_src_ips",
                    "unique_dst_ips",
                    "unique_src_ports",
                    "unique_dst_ports",
                    "flow_count",
                    "avg_packet_size",
                    "src_ip_entropy",
                    "dst_ip_entropy",
                    "src_port_entropy",
                    "dst_port_entropy",
                    "packet_size_entropy",
                    "protocol_entropy",
                    "top10_src_ip_fraction",
                    "top10_dst_ip_fraction"
                ]
            }
    
    def extract_features(self, window_stats: Dict) -> np.ndarray:
        """
        Extract features from window statistics.
        
        Args:
            window_stats: Dictionary containing window statistics
            
        Returns:
            numpy array of features
        """
        features = []
        
        # Basic statistics
        total_packets = window_stats.get('total_packets', 0)
        total_bytes = window_stats.get('total_bytes', 0)
        unique_src_ips = window_stats.get('unique_src_ips', 0)
        unique_dst_ips = window_stats.get('unique_dst_ips', 0)
        flow_count = window_stats.get('flow_count', 0)
        
        # Calculate derived features
        avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0
        
        # Get histogram sizes
        src_ip_counts = window_stats.get('src_ip_counts', {})
        dst_ip_counts = window_stats.get('dst_ip_counts', {})
        src_port_counts = window_stats.get('src_port_counts', {})
        dst_port_counts = window_stats.get('dst_port_counts', {})
        packet_size_counts = window_stats.get('packet_size_counts', {})
        protocol_counts = window_stats.get('protocol_counts', {})
        
        unique_src_ports = len(src_port_counts)
        unique_dst_ports = len(dst_port_counts)
        
        # Entropy features (should be pre-calculated)
        entropy_features = window_stats.get('entropy_features', {})
        src_ip_entropy = entropy_features.get('src_ip_entropy', 0.0)
        dst_ip_entropy = entropy_features.get('dst_ip_entropy', 0.0)
        src_port_entropy = entropy_features.get('src_port_entropy', 0.0)
        dst_port_entropy = entropy_features.get('dst_port_entropy', 0.0)
        packet_size_entropy = entropy_features.get('packet_size_entropy', 0.0)
        protocol_entropy = entropy_features.get('protocol_entropy', 0.0)
        
        # Top-N IP fractions
        top10_src_ip_fraction = self._calculate_top_n_fraction(src_ip_counts, total_packets, 10)
        top10_dst_ip_fraction = self._calculate_top_n_fraction(dst_ip_counts, total_packets, 10)
        
        # Build feature vector
        feature_list = [
            total_packets,
            total_bytes,
            unique_src_ips,
            unique_dst_ips,
            unique_src_ports,
            unique_dst_ports,
            flow_count,
            avg_packet_size,
            src_ip_entropy,
            dst_ip_entropy,
            src_port_entropy,
            dst_port_entropy,
            packet_size_entropy,
            protocol_entropy,
            top10_src_ip_fraction,
            top10_dst_ip_fraction
        ]
        
        return np.array(feature_list, dtype=np.float32)
    
    def _calculate_top_n_fraction(self, counts: Dict, total: int, n: int) -> float:
        """Calculate fraction of traffic from top N sources."""
        if total == 0 or len(counts) == 0:
            return 0.0
        
        # Sort by count (descending)
        sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Sum top N
        top_n_sum = sum(count for _, count in sorted_counts[:n])
        
        return top_n_sum / total
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names."""
        return self.features_spec.get('features', [])

