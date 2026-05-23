from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= float(threshold)).astype(int)
    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "AUROC": safe_auc(y_true, y_score),
        "AP": safe_ap(y_true, y_score),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def select_global_threshold(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric: str = "balanced_accuracy",
) -> Tuple[float, Dict[str, Any]]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    candidates = np.unique(np.r_[0.0, 0.5, 1.0, y_score])
    best_threshold = 0.5
    best_value = -np.inf
    best_metrics: Dict[str, Any] = {}
    for threshold in candidates:
        m = binary_metrics(y_true, y_score, float(threshold))
        value = float(m.get(metric, -np.inf))
        if np.isnan(value):
            continue
        if value > best_value:
            best_value = value
            best_threshold = float(threshold)
            best_metrics = m
    return best_threshold, best_metrics


def per_generator_metrics(
    df: pd.DataFrame,
    y_score: np.ndarray,
    threshold: float,
    split: str = "test",
) -> pd.DataFrame:
    work = df.copy()
    work["score"] = np.asarray(y_score, dtype=float)
    if "split" in work.columns:
        work = work[work["split"] == split].copy()
    rows: List[Dict[str, Any]] = []
    group_col = "subset" if "subset" in work.columns else "generator"
    for subset, part in work.groupby(group_col, sort=True):
        y = part["label"].astype(int).values
        scores = part["score"].values
        labels = sorted(set(y.tolist()))
        row: Dict[str, Any] = {
            "split": split,
            "subset": subset,
            "generator": str(part["generator"].iloc[0]) if "generator" in part.columns and len(part) else str(subset),
            "generator_family": str(part["generator_family"].iloc[0]) if "generator_family" in part.columns and len(part) else "",
            "n_real": int(np.sum(y == 0)),
            "n_fake": int(np.sum(y == 1)),
            "valid": bool(labels == [0, 1]),
        }
        if labels == [0, 1]:
            row.update(binary_metrics(y, scores, threshold))
        else:
            row.update(
                {
                    "n": int(len(y)),
                    "threshold": float(threshold),
                    "accuracy": float("nan"),
                    "balanced_accuracy": float("nan"),
                    "AUROC": float("nan"),
                    "AP": float("nan"),
                    "F1": float("nan"),
                    "tn": 0,
                    "fp": 0,
                    "fn": 0,
                    "tp": 0,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_per_generator(per_gen: pd.DataFrame) -> Dict[str, Any]:
    valid = per_gen[per_gen["valid"] == True].copy()
    if valid.empty:
        return {"valid_generators": 0, "invalid_generators": int(len(per_gen))}
    numeric = ["accuracy", "balanced_accuracy", "AUROC", "AP", "F1"]
    summary: Dict[str, Any] = {
        "valid_generators": int(len(valid)),
        "invalid_generators": int(len(per_gen) - len(valid)),
    }
    for col in numeric:
        summary[f"macro_{col}"] = float(valid[col].mean())
        summary[f"worst_{col}"] = float(valid[col].min())
    worst_idx = valid["balanced_accuracy"].astype(float).idxmin()
    summary["worst_generator_by_balanced_accuracy"] = str(valid.loc[worst_idx, "subset"])
    return summary


def confusion_by_generator_family(per_gen: pd.DataFrame) -> pd.DataFrame:
    if "generator_family" not in per_gen.columns:
        return pd.DataFrame()
    rows = []
    for family, part in per_gen.groupby("generator_family", sort=True):
        rows.append(
            {
                "generator_family": family,
                "n_generators": int(len(part)),
                "tn": int(part["tn"].sum()),
                "fp": int(part["fp"].sum()),
                "fn": int(part["fn"].sum()),
                "tp": int(part["tp"].sum()),
                "macro_balanced_accuracy": float(part["balanced_accuracy"].mean()),
                "macro_AUROC": float(part["AUROC"].mean()),
                "macro_AP": float(part["AP"].mean()),
            }
        )
    return pd.DataFrame(rows)

