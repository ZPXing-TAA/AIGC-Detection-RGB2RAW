from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def fixed_fpr_threshold(real_scores: np.ndarray, target_fpr: float) -> Tuple[float, float]:
    if len(real_scores) == 0:
        return float("nan"), float("nan")
    threshold = float(np.percentile(real_scores, 100.0 * (1.0 - float(target_fpr))))
    real_fpr = float(np.mean(real_scores >= threshold))
    return threshold, real_fpr


def compute_detection_metrics(labels: Iterable[int], scores: Iterable[float], generators: Iterable[str] = None) -> Dict[str, Any]:
    labels_np = np.asarray(list(labels), dtype=np.int64)
    scores_np = np.asarray(list(scores), dtype=np.float64)
    pred = (scores_np >= 0.5).astype(np.int64)

    real_mask = labels_np == 0
    fake_mask = labels_np == 1
    metrics: Dict[str, Any] = {
        "n": int(len(labels_np)),
        "n_real": int(np.sum(real_mask)),
        "n_fake": int(np.sum(fake_mask)),
        "overall_acc": float(accuracy_score(labels_np, pred)) if len(labels_np) else float("nan"),
        "photographic_acc": float(np.mean(pred[real_mask] == 0)) if np.any(real_mask) else float("nan"),
        "generated_acc": float(np.mean(pred[fake_mask] == 1)) if np.any(fake_mask) else float("nan"),
        "AUROC": _safe_auc(labels_np, scores_np),
        "AUPR": _safe_ap(labels_np, scores_np),
    }

    real_scores = scores_np[real_mask]
    fake_scores = scores_np[fake_mask]
    for pct, fpr in [("1", 0.01), ("5", 0.05), ("10", 0.10)]:
        threshold, real_fpr = fixed_fpr_threshold(real_scores, fpr)
        fake_tpr = float(np.mean(fake_scores >= threshold)) if len(fake_scores) else float("nan")
        metrics["threshold_{}fpr".format(pct)] = threshold
        metrics["real_fpr_at_{}fpr".format(pct)] = real_fpr
        metrics["TPR@{}%FPR".format(pct)] = fake_tpr

    if generators is not None:
        gen_np = np.asarray(list(generators))
        per_gen = []
        for gen in sorted(set(gen_np.tolist())):
            mask = (gen_np == gen) & fake_mask
            if not np.any(mask):
                continue
            row = {
                "generator": str(gen),
                "n_fake": int(np.sum(mask)),
                "TPR@0.5": float(np.mean(scores_np[mask] >= 0.5)),
            }
            for pct in ["1", "5", "10"]:
                threshold = metrics["threshold_{}fpr".format(pct)]
                row["TPR@{}%FPR".format(pct)] = float(np.mean(scores_np[mask] >= threshold))
            per_gen.append(row)
        metrics["per_generator_tpr"] = per_gen
        if per_gen:
            metrics["worst_generator_TPR@0.5"] = float(min(x["TPR@0.5"] for x in per_gen))
            metrics["worst_generator"] = str(min(per_gen, key=lambda x: x["TPR@0.5"])["generator"])
    return metrics


def predictions_dataframe(batch_rows: List[Dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(batch_rows)

