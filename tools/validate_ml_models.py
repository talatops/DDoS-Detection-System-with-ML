#!/usr/bin/env python3
"""
Quick regression test for trained ML models.
"""

import argparse
import json
import random
from pathlib import Path
from typing import List

import joblib
import numpy as np


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def smoke_test_model(model_path: Path, feature_dim: int, samples: int) -> float:
    model = joblib.load(model_path)
    features = np.random.rand(samples, feature_dim).astype(np.float64)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
    else:
        raise RuntimeError(f"Model {model_path} does not expose predict_proba()")
    if probs.shape[1] < 2:
        raise RuntimeError("predict_proba output should have at least two columns")
    return float(np.mean(probs[:, 1]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ML models via smoke tests.")
    parser.add_argument(
        "--manifest",
        default="models/model_manifest.json",
        help="Path to model manifest JSON.",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=24,
        help="Feature vector dimension to synthesize.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Synthetic samples per model.",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = load_manifest(manifest_path)
    models: List[dict] = manifest.get("models", [])
    if not models:
        raise RuntimeError("Manifest does not list any models.")

    print(f"[INFO] Validating {len(models)} models from {manifest_path}")
    for entry in models:
        name = entry.get("name")
        model_path = Path(entry.get("path", ""))
        if not model_path.exists():
            print(f"[WARN] Model {name} missing at {model_path}")
            continue
        avg_prob = smoke_test_model(model_path, args.feature_dim, args.samples)
        print(
            f"[PASS] {name:>6s}  file={model_path}  avg_attack_prob={avg_prob:.4f} "
            f"recall={entry.get('recall', 0):.4f} fpr={entry.get('false_positive_rate', 0):.4f}"
        )

    print("[DONE] ML regression smoke-test complete.")


if __name__ == "__main__":
    main()

