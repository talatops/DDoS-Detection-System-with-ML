#!/usr/bin/env python3
"""
Test ML model with a single CSV row or feature vector.
Quick test to verify ML inference is working.
"""

import sys
import os
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.ml.inference_engine import MLInferenceEngine
from src.ml.feature_extractor import FeatureExtractor

def test_with_csv_row(csv_file, row_index=0):
    """Test ML model with a single row from CSV file."""
    print(f"=== Testing ML Model with CSV Row ===")
    print(f"File: {csv_file}")
    print(f"Row: {row_index}")
    
    # Load CSV
    try:
        df = pd.read_csv(csv_file, nrows=row_index+1)
        if len(df) == 0:
            print("ERROR: CSV file is empty")
            return False
        row = df.iloc[row_index]
        print(f"\nLoaded row {row_index} with {len(row)} columns")
    except Exception as e:
        print(f"ERROR: Failed to load CSV: {e}")
        return False
    
    # Extract features (simplified - you'll need to adapt this to your feature extraction)
    print("\nExtracting features...")
    try:
        # This is a placeholder - adapt to your actual feature extraction
        features = extract_features_from_csv_row(row)
        print(f"Extracted {len(features)} features")
        print(f"Features: {features[:5]}... (showing first 5)")
    except Exception as e:
        print(f"ERROR: Feature extraction failed: {e}")
        return False
    
    # Load model
    print("\nLoading ML model...")
    model_path = "models/rf_model.joblib"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Please train the model first: python3 src/ml/train_ml.py")
        return False
    
    try:
        engine = MLInferenceEngine()
        if not engine.loadModel(model_path):
            print("ERROR: Failed to load model")
            return False
        print("Model loaded successfully")
    except Exception as e:
        print(f"ERROR: Model loading failed: {e}")
        return False
    
    # Predict
    print("\nMaking prediction...")
    try:
        score = engine.predict(features)
        print(f"\n=== Results ===")
        print(f"Attack Probability: {score:.4f}")
        print(f"Prediction: {'ATTACK' if score > 0.5 else 'BENIGN'}")
        print(f"Confidence: {abs(score - 0.5) * 200:.1f}%")
        
        if score > 0.5:
            print("\n⚠️  DDoS ATTACK DETECTED!")
        else:
            print("\n✓ Traffic appears BENIGN")
            
        return True
    except Exception as e:
        print(f"ERROR: Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def extract_features_from_csv_row(row):
    """Extract features from a CSV row (simplified version)."""
    # This matches the features in train_ml.py
    features = []
    
    # Basic counts
    total_packets = row.get('Total Fwd Packets', 0) + row.get('Total Backward Packets', 0)
    total_bytes = row.get('Total Length of Fwd Packets', 0) + row.get('Total Length of Bwd Packets', 0)
    
    features.append(float(total_packets))
    features.append(float(total_bytes))
    features.append(1.0)  # unique_src_ips (simplified)
    features.append(1.0)  # unique_dst_ips
    features.append(1.0)  # unique_src_ports
    features.append(1.0)  # unique_dst_ports
    features.append(1.0)  # flow_count
    
    # Derived features
    avg_packet_size = total_bytes / total_packets if total_packets > 0 else 0
    features.append(float(avg_packet_size))
    
    # Entropy features (placeholders - would need actual calculation)
    features.append(0.0)  # src_ip_entropy
    features.append(0.0)  # dst_ip_entropy
    features.append(0.0)  # src_port_entropy
    features.append(0.0)  # dst_port_entropy
    
    # Packet size entropy (approximation)
    packet_size_entropy = row.get('Packet Length Std', 0) / (row.get('Packet Length Mean', 1) + 1e-6)
    features.append(float(packet_size_entropy))
    
    features.append(0.0)  # protocol_entropy
    features.append(1.0)  # top10_src_ip_fraction
    features.append(1.0)  # top10_dst_ip_fraction
    
    return features

def test_with_feature_vector(features):
    """Test ML model with a direct feature vector."""
    print(f"=== Testing ML Model with Feature Vector ===")
    print(f"Features: {features}")
    
    model_path = "models/rf_model.joblib"
    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        return False
    
    try:
        engine = MLInferenceEngine()
        if not engine.loadModel(model_path):
            print("ERROR: Failed to load model")
            return False
        
        score = engine.predict(features)
        print(f"\n=== Results ===")
        print(f"Attack Probability: {score:.4f}")
        print(f"Prediction: {'ATTACK' if score > 0.5 else 'BENIGN'}")
        return True
    except Exception as e:
        print(f"ERROR: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Test with CSV row:")
        print("    python3 test_ml_single.py <csv_file> [row_index]")
        print("")
        print("  Test with feature vector (16 features):")
        print("    python3 test_ml_single.py --features <f1> <f2> ... <f16>")
        print("")
        print("Example:")
        print("  python3 test_ml_single.py data/caida-ddos2007/sample.csv 0")
        sys.exit(1)
    
    if sys.argv[1] == "--features":
        # Test with feature vector
        if len(sys.argv) < 18:
            print("ERROR: Need 16 features")
            sys.exit(1)
        features = [float(x) for x in sys.argv[2:18]]
        success = test_with_feature_vector(features)
    else:
        # Test with CSV row
        csv_file = sys.argv[1]
        row_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        success = test_with_csv_row(csv_file, row_index)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

