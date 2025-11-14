#!/usr/bin/env python3
"""
Dataset preparation script for DDoS detection project.
Processes CSV files and prepares data for ML training.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_caida_csv_files(data_dir="data/caida-ddos2007"):
    """Load all CSV files from CAIDA dataset."""
    csv_files = []
    
    # CSV-01-12 directory
    csv_dir_1 = Path(data_dir) / "CSV-01-12" / "01-12"
    if csv_dir_1.exists():
        for csv_file in csv_dir_1.glob("*.csv"):
            csv_files.append(csv_file)
    
    # CSV-03-11 directory
    csv_dir_2 = Path(data_dir) / "CSV-03-11" / "03-11"
    if csv_dir_2.exists():
        for csv_file in csv_dir_2.glob("*.csv"):
            csv_files.append(csv_file)
    
    return csv_files

def extract_features_from_csv(csv_file, max_rows=None):
    """Extract features from a CSV file."""
    print(f"Loading {csv_file}...")
    try:
        df = pd.read_csv(csv_file, nrows=max_rows)
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Check if Label column exists
        if 'Label' in df.columns:
            print(f"  Labels: {df['Label'].value_counts().to_dict()}")
        
        return df
    except Exception as e:
        print(f"  Error loading {csv_file}: {e}")
        return None

def prepare_training_data(output_dir="data/processed"):
    """Prepare training data from CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    csv_files = load_caida_csv_files()
    print(f"Found {len(csv_files)} CSV files")
    
    all_data = []
    for csv_file in csv_files:
        df = extract_features_from_csv(csv_file, max_rows=10000)  # Limit for testing
        if df is not None:
            # Add source file info
            df['source_file'] = csv_file.name
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        output_file = os.path.join(output_dir, "combined_features.csv")
        combined_df.to_csv(output_file, index=False)
        print(f"\nCombined data saved to {output_file}")
        print(f"Total rows: {len(combined_df)}")
        print(f"Total columns: {len(combined_df.columns)}")
        
        if 'Label' in combined_df.columns:
            print(f"\nLabel distribution:")
            print(combined_df['Label'].value_counts())
    
    return combined_df if all_data else None

if __name__ == "__main__":
    print("=== Dataset Preparation ===")
    prepare_training_data()

