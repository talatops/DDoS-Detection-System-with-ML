#!/usr/bin/env python3
"""
Simple Flask dashboard that summarizes detector output from log files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Any

from flask import Flask, jsonify, render_template, send_file

ROOT_DIR = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
RESULTS_DIR = ROOT_DIR / "results"
MODELS_DIR = ROOT_DIR / "models"

app = Flask(
    __name__,
    static_folder=str(Path(__file__).parent / "static"),
    template_folder=str(Path(__file__).parent / "templates"),
)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def load_metrics_history(limit: int = 120) -> Dict[str, List[Any]]:
    rows = read_csv(LOG_DIR / "metrics.csv")
    rows = rows[-limit:]
    return {
        "timestamp": [r.get("timestamp_ms", "") for r in rows],
        "cpu": [float(r.get("cpu_percent", 0) or 0) for r in rows],
        "gpu": [float(r.get("gpu_percent", 0) or 0) for r in rows],
        "pps": [float(r.get("pps_in", 0) or 0) for r in rows],
        "windows": [int(r.get("windows_processed", 0) or 0) for r in rows],
        "model": rows[-1].get("model", "") if rows else "",
        "latest": rows[-1] if rows else {},
    }


def load_alerts(limit: int = 50) -> List[Dict[str, Any]]:
    rows = read_csv(LOG_DIR / "alerts.csv")
    rows = rows[-limit:]
    alerts = []
    for r in rows:
        alerts.append(
            {
                "timestamp": r.get("timestamp_ms", ""),
                "window_index": int(r.get("window_index", 0) or 0),
                "src_ip": r.get("top_src_ip", ""),
                "entropy_score": float(r.get("entropy_score", 0) or 0),
                "ml_score": float(r.get("ml_score", 0) or 0),
                "combined_score": float(r.get("combined_score", 0) or 0),
                "detector": r.get("detector", ""),
                "model": r.get("model", ""),
            }
        )
    return alerts


def load_kernel_times(limit: int = 50) -> List[Dict[str, Any]]:
    rows = read_csv(LOG_DIR / "kernel_times.csv")
    rows = rows[-limit:]
    return [
        {
            "timestamp": r.get("timestamp_ms", ""),
            "kernel": r.get("kernel_name", ""),
            "duration_ms": float(r.get("execution_time_ms", 0) or 0),
        }
        for r in rows
    ]


def load_blocking_events(limit: int = 500) -> List[Dict[str, Any]]:
    rows = read_csv(LOG_DIR / "blocking.csv")
    rows = rows[-limit:]
    return [
        {
            "timestamp": r.get("timestamp_ms", ""),
            "ip": r.get("ip", ""),
            "impacted_packets": int(r.get("impacted_packets", 0) or 0),
            "dropped_packets": int(r.get("dropped_packets", 0) or 0),
        }
        for r in rows
    ]


def load_training_metrics() -> Dict[str, Any]:
    path = REPORTS_DIR / "training_metrics.json"
    if not path.exists():
        return {}
    with path.open() as f:
        data = json.load(f)
    selected = data.get("selected_model")
    selected_model = {}
    for m in data.get("models", []):
        if m.get("name") == selected:
            selected_model = m
            break
    return {
        "timestamp": data.get("timestamp"),
        "dataset": data.get("dataset", {}),
        "selected_model": selected,
        "model_metrics": selected_model.get("metrics", {}),
        "model_info": selected_model,
    }


def load_model_manifest() -> Dict[str, Any]:
    path = MODELS_DIR / "model_manifest.json"
    if not path.exists():
        return {}
    with path.open() as f:
        return json.load(f)


def build_dashboard_payload() -> Dict[str, Any]:
    metrics = load_metrics_history()
    training = load_training_metrics()
    manifest = load_model_manifest()

    latest_metrics = metrics.get("latest", {})
    summary = {
        "cpu_percent": float(latest_metrics.get("cpu_percent", 0) or 0),
        "gpu_percent": float(latest_metrics.get("gpu_percent", 0) or 0),
        "memory_mb": float(latest_metrics.get("memory_mb", 0) or 0),
        "pps_in": float(latest_metrics.get("pps_in", 0) or 0),
        "windows_processed": int(latest_metrics.get("windows_processed", 0) or 0),
        "model": latest_metrics.get("model", manifest.get("selected_model", "unknown")),
    }

    return {
        "summary": summary,
        "metrics_history": {
            "timestamp": metrics.get("timestamp", []),
            "cpu": metrics.get("cpu", []),
            "gpu": metrics.get("gpu", []),
            "pps": metrics.get("pps", []),
        },
        "alerts": load_alerts(),
        "kernel_times": load_kernel_times(),
        "blocking": load_blocking_events(),
        "training": training,
        "model_manifest": manifest,
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/dashboard")
def api_dashboard():
    return jsonify(build_dashboard_payload())


@app.route("/api/training-report")
def api_training_report():
    report_path = REPORTS_DIR / "training_report.txt"
    if not report_path.exists():
        return jsonify({"error": "training report not found"}), 404
    with report_path.open() as f:
        content = f.read()
    return content, 200, {"Content-Type": "text/plain"}


@app.route("/api/roc-curve")
def api_roc_curve():
    roc_path = RESULTS_DIR / "ml_roc_curve.png"
    if not roc_path.exists():
        return jsonify({"error": "roc curve not found"}), 404
    return send_file(roc_path, mimetype="image/png")


if __name__ == "__main__":
    print("=== Dashboard available at http://0.0.0.0:5002 ===")
    app.run(host="0.0.0.0", port=5002, debug=False)

