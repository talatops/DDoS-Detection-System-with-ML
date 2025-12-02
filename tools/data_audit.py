#!/usr/bin/env python3
"""
Audit training datasets and generate metadata for ROC/PR analysis.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ATTACK_KEYWORDS = [
    "ddos",
    "drdos",
    "syn",
    "udp",
    "ldap",
    "mssql",
    "netbios",
    "ssdp",
    "dns",
    "snmp",
    "ntp",
    "portmap",
    "udplag",
]


def detect_label_column(columns: List[str]) -> Optional[str]:
    for col in columns:
        if col.strip().lower() == "label":
            return col
    return None


def infer_label_from_filename(filename: str) -> int:
    lower_name = filename.lower()
    return int(any(keyword in lower_name for keyword in ATTACK_KEYWORDS))


def summarize_csv(csv_path: Path, max_rows: int) -> Dict:
    try:
        df = pd.read_csv(csv_path, nrows=max_rows, low_memory=False)
    except Exception as exc:  # pylint: disable=broad-except
        return {
            "file": str(csv_path),
            "rows_scanned": 0,
            "attack_rows": 0,
            "benign_rows": 0,
            "error": str(exc),
        }

    label_col = detect_label_column(df.columns)
    if label_col:
        labels = (df[label_col] != "BENIGN").astype(int)
    else:
        inferred = infer_label_from_filename(csv_path.name)
        labels = pd.Series([inferred] * len(df))

    attack_rows = int(labels.sum())
    benign_rows = int(len(labels) - attack_rows)
    return {
        "file": str(csv_path),
        "rows_scanned": int(len(df)),
        "attack_rows": attack_rows,
        "benign_rows": benign_rows,
        "label_column": label_col or "inferred",
    }


def summarize_pcap(dataset_path: Path) -> Dict:
    sizes = [f.stat().st_size for f in dataset_path.rglob("*.pcap")]
    return {
        "pcap_files": len(sizes),
        "total_bytes": sum(sizes),
        "largest_file_bytes": max(sizes) if sizes else 0,
    }


def audit_dataset(dataset_path: Path, max_files: int, max_rows: int) -> Dict:
    csv_files = sorted(dataset_path.rglob("*.csv"))
    csv_stats: List[Dict] = []
    total_rows = 0
    total_attack = 0
    for csv_path in csv_files[:max_files]:
        stats = summarize_csv(csv_path, max_rows)
        csv_stats.append(stats)
        total_rows += stats["rows_scanned"]
        total_attack += stats["attack_rows"]

    benign_rows = total_rows - total_attack
    suggested_train = int(total_rows * 0.8)
    suggested_test = total_rows - suggested_train

    summary = {
        "name": dataset_path.name,
        "path": str(dataset_path),
        "csv_files_discovered": len(csv_files),
        "csv_files_scanned": len(csv_stats),
        "rows_scanned": total_rows,
        "attack_rows": total_attack,
        "benign_rows": benign_rows,
        "suggested_split": {
            "train_rows": suggested_train,
            "test_rows": suggested_test,
        },
        "csv_samples": csv_stats,
        "pcap_summary": summarize_pcap(dataset_path),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit datasets for training metadata.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data/caida-ddos2007", "data/cic-ddos2019"],
        help="Dataset directories to audit.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=10,
        help="Maximum CSV files to scan per dataset.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=50000,
        help="Maximum rows to read per CSV file.",
    )
    parser.add_argument(
        "--output",
        default="data/dataset_metadata.json",
        help="Path to write dataset metadata JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets: List[Dict] = []
    for dataset in args.datasets:
        path = Path(dataset)
        if not path.exists():
            print(f"[WARN] Dataset path not found: {path}")
            continue
        summary = audit_dataset(path, args.max_files, args.max_rows)
        datasets.append(summary)
        print(
            f"[INFO] {summary['name']}: rows={summary['rows_scanned']:,} "
            f"attacks={summary['attack_rows']:,} "
            f"csv_scanned={summary['csv_files_scanned']}/{summary['csv_files_discovered']}"
        )

    metadata = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "max_rows_per_file": args.max_rows,
        "datasets": datasets,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"[INFO] Dataset metadata written to {output_path}")


if __name__ == "__main__":
    main()

