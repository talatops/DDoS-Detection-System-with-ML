#!/usr/bin/env python3
"""
Train multiple ML models (RF, GBDT, DNN) and export manifest/metrics.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.ml import train_ml  # type: ignore  # pylint: disable=wrong-import-position
from src.ml.model_wrappers import TorchModelWrapper  # type: ignore  # pylint: disable=wrong-import-position
from src.ml.preprocessor import FeaturePreprocessor  # type: ignore  # pylint: disable=wrong-import-position


try:  # Optional dependency
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    torch = None
    nn = None
    DataLoader = None


@dataclass
class ModelResult:
    name: str
    algorithm: str
    model_obj: object
    metrics: Dict
    artifact_path: Path
    preprocessor_path: Path
    imports: List[str] = field(default_factory=list)
    extra_evals: Dict[str, Dict[str, float]] = field(default_factory=dict)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ensemble of ML models for DDoS detection.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["data/caida-ddos2007"],
        help="Dataset directories containing CSV files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["rf", "gbdt", "dnn"],
        choices=["rf", "gbdt", "dnn"],
        help="Models to train.",
    )
    parser.add_argument("--max-csv-rows", type=int, default=-1, help="Rows per CSV during loading (-1 for all rows).")
    parser.add_argument("--test-size", type=float, default=0.3, help="Test split fraction.")
    parser.add_argument("--output-dir", default="models", help="Directory for trained artifacts.")
    parser.add_argument(
        "--balance-data",
        action="store_true",
        help="Downsample majority class to balance training data (reduces class imbalance warnings).",
    )
    parser.add_argument(
        "--max-ratio",
        type=float,
        default=5.0,
        help="Maximum ratio of majority to minority class after balancing (default: 5.0).",
    )
    parser.add_argument(
        "--kfolds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (applies to scikit-learn models).",
    )
    parser.add_argument(
        "--eval-datasets",
        nargs="+",
        default=[],
        help="Additional CSV directories used for post-training evaluation.",
    )
    return parser.parse_args()


def combine_datasets(
    paths: List[str], max_rows: int, balance: bool = False, max_ratio: float = 5.0
) -> Tuple[pd.DataFrame, List[Path]]:
    frames = []
    files: List[Path] = []
    for dataset in paths:
        limit = max_rows if max_rows and max_rows > 0 else None
        df, csv_files = train_ml.load_training_data(dataset, max_rows=limit)
        if df is None:
            continue
        frames.append(df)
        files.extend(csv_files or [])
    if not frames:
        raise RuntimeError("No datasets could be loaded.")
    combined = pd.concat(frames, ignore_index=True)
    
    # Balance data if requested
    if balance and "label" in combined.columns:
        attack_count = combined["label"].sum()
        benign_count = len(combined) - attack_count
        
        if attack_count > 0 and benign_count > 0:
            ratio = max(attack_count, benign_count) / min(attack_count, benign_count)
            if ratio > max_ratio:
                minority_class = 0 if attack_count > benign_count else 1
                minority_count = min(attack_count, benign_count)
                target_majority = int(minority_count * max_ratio)
                
                majority_mask = combined["label"] == (1 - minority_class)
                majority_df = combined[majority_mask]
                if len(majority_df) > target_majority:
                    majority_df = majority_df.sample(n=target_majority, random_state=42)
                
                minority_df = combined[combined["label"] == minority_class]
                combined = pd.concat([majority_df, minority_df], ignore_index=True).sample(
                    frac=1, random_state=42
                ).reset_index(drop=True)
                
                new_attack = combined["label"].sum()
                new_benign = len(combined) - new_attack
                print(
                    f"[BALANCE] Original: {attack_count} attacks, {benign_count} benign "
                    f"(ratio {ratio:.1f}:1)"
                )
                print(
                    f"[BALANCE] Balanced: {new_attack} attacks, {new_benign} benign "
                    f"(ratio {max(new_attack, new_benign) / min(new_attack, new_benign):.1f}:1)"
                )
    
    return combined, files


def preprocess_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, FeaturePreprocessor]:
    X = train_ml.extract_features_from_csv(df)
    y = df["label"].values
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    preprocessor = FeaturePreprocessor(
        use_scaling=False,
        use_outlier_removal=True,
        use_feature_selection=False,
        n_features=min(24, X.shape[1]),
    )
    X_processed = preprocessor.fit_transform(X, y)
    X_array = np.asarray(X_processed, dtype=np.float32)
    X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
    return X_array, y, preprocessor


def _predict_labels(model, X) -> np.ndarray:
    if hasattr(model, "predict"):
        return model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    return (probs >= 0.5).astype(int)


def _score_classifier(model, X, y) -> float:
    preds = _predict_labels(model, X)
    return float(np.mean(preds == y))


def evaluate_classifier(
    model, X_train, y_train, X_test, y_test, name: str, kfolds: int = 5
) -> Dict:
    if hasattr(model, "score") and hasattr(model, "predict"):
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        cv_splits = max(2, kfolds)
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv_splits, scoring="f1")
    else:
        train_score = _score_classifier(model, X_train, y_train)
        test_score = _score_classifier(model, X_test, y_test)
        cv_scores = np.array([0.0])

    y_pred = _predict_labels(model, X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    roc_auc = auc(fpr_curve, tpr_curve)
    pr_auc = auc(recall_curve, precision_curve)

    return {
        "train_accuracy": float(train_score),
        "test_accuracy": float(test_score),
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "precision": float(precision),
        "recall": float(recall),
        "false_positive_rate": float(fpr),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Benign", "Attack"], digits=4
        ),
    }


def evaluate_on_dataset(model, X, y) -> Dict[str, float]:
    if len(y) == 0:
        return {}
    y_pred = _predict_labels(model, X)
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)[:, 1]
        except Exception:
            y_proba = np.zeros_like(y_pred, dtype=float)
    else:
        y_proba = np.zeros_like(y_pred, dtype=float)
    cm = confusion_matrix(y, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / max(1, (tp + tn + fp + fn))
    f1 = (
        (2 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    try:
        fpr_curve, tpr_curve, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr_curve, tpr_curve)
        precision_curve, recall_curve, _ = precision_recall_curve(y, y_proba)
        pr_auc = auc(recall_curve, precision_curve)
    except ValueError:
        roc_auc = 0.0
        pr_auc = 0.0
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "false_positive_rate": float(fp / (fp + tn) if (fp + tn) > 0 else 0.0),
    }


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=256,
        max_depth=18,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_gbdt(X_train, y_train) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(
        max_depth=12,
        learning_rate=0.08,
        max_iter=400,
        l2_regularization=1e-3,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def _torch_device() -> Optional["torch.device"]:
    if torch is None:
        return None
    if torch.cuda.is_available():
        try:
            device = torch.device("cuda")
            device_name = torch.cuda.get_device_name(device)
            print(f"[DNN] Using CUDA device: {device_name}")
            return device
        except Exception as exc:  # pylint: disable=broad-except
            print(f"[DNN] CUDA reported but unavailable ({exc}); falling back to CPU.")
    print("[DNN] Using CPU for DNN training.")
    return torch.device("cpu") if torch is not None else None


def train_dnn(X_train, y_train, epochs: int = 5, batch_size: int = 512):
    if torch is None or nn is None or DataLoader is None:
        raise RuntimeError("PyTorch is required for DNN training.")
    input_dim = X_train.shape[1]
    hidden_sizes = [128, 64, 32]
    device = _torch_device()
    if device is None:
        raise RuntimeError("PyTorch device unavailable.")

    class _Net(nn.Module):
        def __init__(self, input_size):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], 1),
            )

        def forward(self, x):
            return self.network(x)

    net = _Net(input_dim).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()
    dataset = TensorDataset(
        torch.from_numpy(X_train).float(),
        torch.from_numpy(y_train).float().view(-1, 1),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    net.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            logits = net(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
    net.eval()
    state_dict = {k: v.cpu() for k, v in net.state_dict().items()}
    wrapper = TorchModelWrapper(
        input_dim=input_dim,
        hidden_sizes=hidden_sizes,
        state_dict=state_dict,
    )
    return wrapper


def save_manifest(results: List[ModelResult], output_path: Path, selected: str) -> None:
    """
    Persist model metadata to model_manifest.json.

    Behaviour:
    - If a manifest already exists, keep entries for models that were NOT
      trained in this run and update/overwrite entries for models that WERE.
    - This lets you train models one-by-one (rf, gbdt, dnn) while keeping
      historical details for the other models.
    - On subsequent retrains of a given model, only that model's entry is
      refreshed; others are left untouched.
    """
    existing: Dict = {}
    if output_path.exists():
        try:
            with output_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
        except Exception:  # pragma: no cover - best-effort merge
            existing = {}

    # Index existing models by name so we can update in-place
    existing_models: Dict[str, Dict] = {}
    for model in existing.get("models", []):
        name = model.get("name")
        if isinstance(name, str):
            existing_models[name] = model

    # Update/insert entries for models trained in this run
    for res in results:
        existing_models[res.name] = {
            "name": res.name,
            "type": res.algorithm,
            "path": str(res.artifact_path),
            "preprocessor": str(res.preprocessor_path),
            "imports": res.imports,
            "recall": res.metrics.get("recall", 0.0),
            "false_positive_rate": res.metrics.get("false_positive_rate", 0.0),
            "roc_auc": res.metrics.get("roc_auc", 0.0),
        }

    # Preserve any previous default model if present, otherwise fall back
    # to the first model trained in this run.
    default_model = existing.get("default_model")
    if not default_model and results:
        default_model = results[0].name

    # The "selected_model" reflects the best model from THIS run (previous
    # behaviour), but the manifest still contains all other models.
    selected_model = selected or existing.get("selected_model") or default_model or ""

    manifest = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "default_model": default_model or "",
        "selected_model": selected_model,
        # Sort by name for stable presentation in the dashboard table
        "models": [existing_models[name] for name in sorted(existing_models.keys())],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def write_reports(
    dataset_info: Dict,
    results: List[ModelResult],
    selected: str,
    evaluation: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    report_path = reports_dir / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("=" * 80 + "\n")
        handle.write("DDoS Detection Model Training Bundle\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Generated: {timestamp}\n\n")
        handle.write("DATASETS\n--------\n")
        handle.write(json.dumps(dataset_info, indent=2))
        handle.write("\n\nMODEL RESULTS\n-------------\n")
        for res in results:
            handle.write(f"* {res.name} ({res.algorithm})\n")
            handle.write(f"  Recall: {res.metrics['recall']:.4f}\n")
            handle.write(f"  FPR: {res.metrics['false_positive_rate']:.4f}\n")
            handle.write(f"  ROC AUC: {res.metrics['roc_auc']:.4f}\n")
            handle.write(f"  Artifact: {res.artifact_path}\n\n")
        handle.write(f"Selected model: {selected}\n")
        if evaluation:
            handle.write("\nADDITIONAL EVALUATIONS\n----------------------\n")
            for dataset_path, metrics_map in evaluation.items():
                handle.write(f"{dataset_path}:\n")
                for model_name, eval_metrics in metrics_map.items():
                    handle.write(
                        f"  - {model_name}: "
                        f"accuracy {eval_metrics.get('accuracy', 0):.4f}, "
                        f"recall {eval_metrics.get('recall', 0):.4f}, "
                        f"precision {eval_metrics.get('precision', 0):.4f}, "
                        f"ROC AUC {eval_metrics.get('roc_auc', 0):.4f}\n"
                    )

    metrics_json = {
        "timestamp": timestamp,
        "dataset": dataset_info,
        "selected_model": selected,
        "models": [
            {
                "name": res.name,
                "algorithm": res.algorithm,
                "metrics": res.metrics,
                "artifact": str(res.artifact_path),
                "preprocessor": str(res.preprocessor_path),
            }
            for res in results
        ],
        "evaluation": evaluation,
    }
    with open(reports_dir / "training_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics_json, handle, indent=2)

    selection_report = reports_dir / "model_selection.md"
    with open(selection_report, "w", encoding="utf-8") as handle:
        handle.write("# Model Selection Summary\n\n")
        handle.write(f"*Generated {timestamp} UTC*\n\n")
        for res in results:
            handle.write(
                f"- **{res.name}** ({res.algorithm}) — recall {res.metrics['recall']:.4f}, "
                f"FPR {res.metrics['false_positive_rate']:.4f}, ROC AUC {res.metrics['roc_auc']:.4f}\n"
            )
        handle.write(f"\n**Selected model:** `{selected}`\n")


def select_best_model(results: List[ModelResult]) -> str:
    if not results:
        return ""
    scored = sorted(
        results,
        key=lambda r: (r.metrics.get("recall", 0.0) - r.metrics.get("false_positive_rate", 0.0) * 0.5),
        reverse=True,
    )
    return scored[0].name


def main() -> None:
    args = parse_args()
    df, csv_files = combine_datasets(
        args.datasets, args.max_csv_rows, args.balance_data, args.max_ratio
    )
    X, y, preprocessor = preprocess_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    models_dir = Path(args.output_dir)
    models_dir.mkdir(exist_ok=True)
    preprocessor_path = models_dir / "preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_path)

    results: List[ModelResult] = []

    if "rf" in args.models:
        rf = train_random_forest(X_train, y_train)
        metrics = evaluate_classifier(
            rf, X_train, y_train, X_test, y_test, "rf", kfolds=args.kfolds
        )
        artifact = models_dir / "rf_model.joblib"
        joblib.dump(rf, artifact)
        results.append(
            ModelResult(
                name="rf",
                algorithm="random_forest",
                model_obj=rf,
                metrics=metrics,
                artifact_path=artifact,
                preprocessor_path=preprocessor_path,
                imports=["preprocessor"],
            )
        )

    if "gbdt" in args.models:
        gbdt = train_gbdt(X_train, y_train)
        metrics = evaluate_classifier(
            gbdt, X_train, y_train, X_test, y_test, "gbdt", kfolds=args.kfolds
        )
        artifact = models_dir / "gbdt_model.joblib"
        joblib.dump(gbdt, artifact)
        results.append(
            ModelResult(
                name="gbdt",
                algorithm="hist_gradient_boosting",
                model_obj=gbdt,
                metrics=metrics,
                artifact_path=artifact,
                preprocessor_path=preprocessor_path,
                imports=["preprocessor"],
            )
        )

    if "dnn" in args.models:
        if torch is None:
            print("[WARN] Skipping DNN training because PyTorch is unavailable.")
        else:
            dnn_wrapper = train_dnn(X_train, y_train)
            metrics = evaluate_classifier(
                dnn_wrapper, X_train, y_train, X_test, y_test, "dnn", kfolds=args.kfolds
            )
            artifact = models_dir / "dnn_model.joblib"
            joblib.dump(dnn_wrapper, artifact)
            results.append(
                ModelResult(
                    name="dnn",
                    algorithm="torch_mlp",
                    model_obj=dnn_wrapper,
                    metrics=metrics,
                    artifact_path=artifact,
                    preprocessor_path=preprocessor_path,
                    imports=["preprocessor", "model_wrappers"],
                )
            )

    if not results:
        raise RuntimeError("No models were trained. Specify at least one via --models.")

    evaluation_summaries: Dict[str, Dict[str, Dict[str, float]]] = {}
    if args.eval_datasets:
        limit = args.max_csv_rows if args.max_csv_rows and args.max_csv_rows > 0 else None
        for eval_path in args.eval_datasets:
            eval_df, _ = train_ml.load_training_data(eval_path, max_rows=limit)
            if eval_df is None:
                print(f"[EVAL] Skipping {eval_path}: no CSV files found.")
                continue
            try:
                X_eval = train_ml.extract_features_from_csv(eval_df)
                if not isinstance(X_eval, pd.DataFrame):
                    X_eval = pd.DataFrame(X_eval)
                X_eval_proc = preprocessor.transform(X_eval)
                X_eval_proc = np.asarray(X_eval_proc)
                y_eval = eval_df["label"].values
            except Exception as exc:
                print(f"[EVAL] Failed to process {eval_path}: {exc}")
                continue
            evaluation_summaries[eval_path] = {}
            for res in results:
                metrics = evaluate_on_dataset(res.model_obj, X_eval_proc, y_eval)
                res.extra_evals[eval_path] = metrics
                evaluation_summaries[eval_path][res.name] = metrics

    # Calculate class distribution
    attack_count = int(y.sum())
    benign_count = int(len(y) - attack_count)
    train_attack = int(y_train.sum())
    train_benign = int(len(y_train) - train_attack)
    test_attack = int(y_test.sum())
    test_benign = int(len(y_test) - test_attack)
    
    dataset_info = {
        "csv_files": [str(path) for path in csv_files],
        "total_rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": int(X.shape[1]),
        "class_distribution": {
            "total": {"attacks": attack_count, "benign": benign_count},
            "train": {"attacks": train_attack, "benign": train_benign},
            "test": {"attacks": test_attack, "benign": test_benign},
        },
        "balanced": args.balance_data,
        "max_ratio": args.max_ratio if args.balance_data else None,
        "parameters": {
            "test_size": args.test_size,
            "kfolds": args.kfolds,
            "max_csv_rows": args.max_csv_rows,
            "eval_datasets": args.eval_datasets,
        },
    }

    selected = select_best_model(results)
    save_manifest(results, Path("models/model_manifest.json"), selected)
    write_reports(dataset_info, results, selected, evaluation_summaries)
    print(f"[DONE] Trained {len(results)} models. Selected '{selected}'.")


if __name__ == "__main__":
    main()

