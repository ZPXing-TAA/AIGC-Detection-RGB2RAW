#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis_utils import feature_columns, feature_group_columns
from src.config_utils import copy_config, ensure_dir, load_config, resolve_path, update_run_metadata, write_json


METRIC_COLUMNS = ["accuracy", "balanced_accuracy", "auroc", "average_precision"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate AIGIBench handcrafted RAW-like features.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--features", required=True, help="Feature parquet or CSV path.")
    parser.add_argument("--output-dir", default=None, help="Override aigibench_eval.output_dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing evaluation outputs.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Mark outputs as a smoke run. Metrics are only for pipeline validation.",
    )
    return parser.parse_args()


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def matrix_from_df(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    return df.loc[:, list(cols)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)


def labels_from_df(df: pd.DataFrame) -> np.ndarray:
    return df["label"].to_numpy(dtype=np.int64)


def finite_float(value) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return out


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> Dict[str, Optional[float]]:
    if len(y_true) == 0:
        return {key: None for key in METRIC_COLUMNS}
    preds = (scores >= threshold).astype(np.int64)
    out: Dict[str, Optional[float]] = {
        "accuracy": finite_float(accuracy_score(y_true, preds)),
        "balanced_accuracy": finite_float(balanced_accuracy_score(y_true, preds)),
    }
    if len(np.unique(y_true)) > 1:
        out["auroc"] = finite_float(roc_auc_score(y_true, scores))
        out["average_precision"] = finite_float(average_precision_score(y_true, scores))
    else:
        out["auroc"] = None
        out["average_precision"] = None
    return out


def select_global_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Dict[str, Optional[float]]]:
    if len(y_true) == 0:
        return 0.5, {key: None for key in METRIC_COLUMNS}
    unique_scores = np.unique(scores[np.isfinite(scores)])
    if unique_scores.size == 0:
        return 0.5, {key: None for key in METRIC_COLUMNS}
    eps = max(float(np.std(unique_scores)) * 1e-6, 1e-12)
    candidates = np.concatenate(
        [
            np.asarray([float(unique_scores.min() - eps)]),
            unique_scores,
            np.asarray([float(unique_scores.max() + eps)]),
        ]
    )
    best_threshold = float(candidates[0])
    best_metrics = binary_metrics(y_true, scores, best_threshold)
    best_score = best_metrics.get("balanced_accuracy")
    best_score = -1.0 if best_score is None else float(best_score)
    for threshold in candidates[1:]:
        metrics = binary_metrics(y_true, scores, float(threshold))
        metric_value = metrics.get("balanced_accuracy")
        metric_score = -1.0 if metric_value is None else float(metric_value)
        if metric_score > best_score:
            best_score = metric_score
            best_threshold = float(threshold)
            best_metrics = metrics
    return best_threshold, best_metrics


class LogisticProtocol:
    def __init__(self, max_iter: int, class_weight: str, seed: int):
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.model = LogisticRegression(max_iter=max_iter, class_weight=class_weight, random_state=seed)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_imp = self.imputer.fit_transform(x_train)
        x_scaled = self.scaler.fit_transform(x_imp)
        self.model.fit(x_scaled, y_train)

    def scores(self, x: np.ndarray) -> np.ndarray:
        x_imp = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imp)
        return self.model.predict_proba(x_scaled)[:, 1].astype(np.float64)


class GMMProtocol:
    def __init__(self, components: int, pca_components: int, covariance_type: str, reg_covar: float, seed: int):
        self.components = int(components)
        self.pca_components = int(pca_components)
        self.covariance_type = covariance_type
        self.reg_covar = float(reg_covar)
        self.seed = int(seed)
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.pca: Optional[PCA] = None
        self.gmm: Optional[GaussianMixture] = None
        self.n_components_fit = 0
        self.pca_components_fit = 0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> None:
        x_photo = x_train[y_train == 0]
        if len(x_photo) < 2:
            raise RuntimeError("GMM requires at least two train real/photo samples.")
        x_imp = self.imputer.fit_transform(x_photo)
        x_scaled = self.scaler.fit_transform(x_imp)
        max_pca = min(int(self.pca_components), x_scaled.shape[0], x_scaled.shape[1])
        if max_pca < 1:
            raise RuntimeError("GMM PCA has no valid components.")
        self.pca = PCA(n_components=max_pca, random_state=self.seed)
        x_pca = self.pca.fit_transform(x_scaled)
        self.pca_components_fit = int(max_pca)
        self.n_components_fit = max(1, min(int(self.components), x_pca.shape[0]))
        self.gmm = GaussianMixture(
            n_components=self.n_components_fit,
            covariance_type=self.covariance_type,
            reg_covar=self.reg_covar,
            random_state=self.seed,
        )
        self.gmm.fit(x_pca)

    def scores(self, x: np.ndarray) -> np.ndarray:
        if self.pca is None or self.gmm is None:
            raise RuntimeError("GMMProtocol must be fit before scoring.")
        x_imp = self.imputer.transform(x)
        x_scaled = self.scaler.transform(x_imp)
        x_pca = self.pca.transform(x_scaled)
        return (-self.gmm.score_samples(x_pca)).astype(np.float64)


def summarize_generator_metrics(df_metrics: pd.DataFrame) -> Dict[str, object]:
    valid = df_metrics[df_metrics["valid"] == True].copy()
    summary: Dict[str, object] = {
        "num_generators_total": int(len(df_metrics)),
        "num_generators_valid": int(len(valid)),
        "num_generators_skipped": int(len(df_metrics) - len(valid)),
    }
    if valid.empty:
        summary["macro_average"] = {key: None for key in METRIC_COLUMNS}
        summary["worst_generator"] = None
        return summary
    summary["macro_average"] = {
        key: finite_float(valid[key].mean()) for key in METRIC_COLUMNS if key in valid.columns
    }
    worst = valid.sort_values(["balanced_accuracy", "generator"], ascending=[True, True]).iloc[0]
    summary["worst_generator"] = {
        "generator": str(worst["generator"]),
        "balanced_accuracy": finite_float(worst["balanced_accuracy"]),
        "accuracy": finite_float(worst["accuracy"]),
        "auroc": finite_float(worst["auroc"]),
        "average_precision": finite_float(worst["average_precision"]),
        "num_real": int(worst["num_real"]),
        "num_fake": int(worst["num_fake"]),
    }
    return summary


def per_generator_metrics(df_test: pd.DataFrame, scores: np.ndarray, threshold: float, method: str) -> pd.DataFrame:
    eval_df = df_test.reset_index(drop=True).copy()
    eval_df["_score"] = scores
    records: List[Dict[str, object]] = []
    for generator, group in eval_df.groupby("generator", sort=True):
        y = labels_from_df(group)
        group_scores = group["_score"].to_numpy(dtype=np.float64)
        num_real = int(np.sum(y == 0))
        num_fake = int(np.sum(y == 1))
        record: Dict[str, object] = {
            "method": method,
            "generator": generator,
            "generator_family": str(group["generator_family"].iloc[0]) if "generator_family" in group else "",
            "num_images": int(len(group)),
            "num_real": num_real,
            "num_fake": num_fake,
            "threshold": float(threshold),
            "valid": bool(num_real > 0 and num_fake > 0),
            "skip_reason": "",
        }
        if not record["valid"]:
            record["skip_reason"] = "missing_real_or_fake"
            for key in METRIC_COLUMNS:
                record[key] = None
        else:
            record.update(binary_metrics(y, group_scores, threshold))
        records.append(record)
    return pd.DataFrame(records)


def confusion_by_family(df_test: pd.DataFrame, scores: np.ndarray, threshold: float, method: str) -> List[Dict[str, object]]:
    eval_df = df_test.reset_index(drop=True).copy()
    eval_df["_score"] = scores
    eval_df["_pred"] = (scores >= threshold).astype(np.int64)
    records: List[Dict[str, object]] = []
    for family, group in eval_df.groupby("generator_family", sort=True):
        y = labels_from_df(group)
        pred = group["_pred"].to_numpy(dtype=np.int64)
        labels = [0, 1]
        cm = confusion_matrix(y, pred, labels=labels)
        tn, fp, fn, tp = [int(v) for v in cm.ravel()]
        records.append(
            {
                "method": method,
                "generator_family": family,
                "num_images": int(len(group)),
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "accuracy": finite_float(accuracy_score(y, pred)),
                "balanced_accuracy": finite_float(balanced_accuracy_score(y, pred)) if len(np.unique(y)) > 1 else None,
            }
        )
    return records


def run_lr_protocol(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: Sequence[str],
    cfg: dict,
    seed: int,
) -> Dict[str, object]:
    lr_cfg = cfg.get("aigibench_eval", {}).get("logistic_regression", {})
    protocol = LogisticProtocol(
        max_iter=int(lr_cfg.get("max_iter", 2000)),
        class_weight=str(lr_cfg.get("class_weight", "balanced")),
        seed=seed,
    )
    x_train = matrix_from_df(train_df, cols)
    y_train = labels_from_df(train_df)
    protocol.fit(x_train, y_train)
    val_scores = protocol.scores(matrix_from_df(val_df, cols))
    threshold, val_metrics = select_global_threshold(labels_from_df(val_df), val_scores)
    test_scores = protocol.scores(matrix_from_df(test_df, cols))
    per_gen = per_generator_metrics(test_df, test_scores, threshold, method="logistic_regression")
    return {
        "protocol": protocol,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_scores": test_scores,
        "per_generator": per_gen,
        "summary": summarize_generator_metrics(per_gen),
    }


def run_gmm_protocol(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: Sequence[str],
    cfg: dict,
    seed: int,
) -> Dict[str, object]:
    gmm_cfg = cfg.get("aigibench_eval", {}).get("gmm", {})
    protocol = GMMProtocol(
        components=int(gmm_cfg.get("components", cfg.get("analysis", {}).get("gmm_components", 5))),
        pca_components=int(gmm_cfg.get("pca_components", 32)),
        covariance_type=str(gmm_cfg.get("covariance_type", "full")),
        reg_covar=float(gmm_cfg.get("reg_covar", 1e-5)),
        seed=seed,
    )
    x_train = matrix_from_df(train_df, cols)
    y_train = labels_from_df(train_df)
    protocol.fit(x_train, y_train)
    val_scores = protocol.scores(matrix_from_df(val_df, cols))
    threshold, val_metrics = select_global_threshold(labels_from_df(val_df), val_scores)
    test_scores = protocol.scores(matrix_from_df(test_df, cols))
    per_gen = per_generator_metrics(test_df, test_scores, threshold, method="photo_gmm")
    return {
        "protocol": protocol,
        "threshold": threshold,
        "val_metrics": val_metrics,
        "test_scores": test_scores,
        "per_generator": per_gen,
        "summary": summarize_generator_metrics(per_gen),
    }


def write_dataframe_outputs(df: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    write_json(json_path, {"records": records})


def save_gmm_likelihood_hist(df_test: pd.DataFrame, anomaly_scores: np.ndarray, path: Path) -> None:
    ensure_dir(path.parent)
    log_likelihood = -np.asarray(anomaly_scores, dtype=np.float64)
    labels = labels_from_df(df_test.reset_index(drop=True))
    plt.figure(figsize=(7, 4))
    plt.hist(log_likelihood[labels == 0], bins=50, alpha=0.65, density=True, label="real", color="#2f6f9f")
    plt.hist(log_likelihood[labels == 1], bins=50, alpha=0.65, density=True, label="fake", color="#c44e52")
    plt.title("AIGIBench photo-trained GMM log likelihood")
    plt.xlabel("log likelihood")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_feature_group_ablation(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cols: Sequence[str],
    cfg: dict,
    seed: int,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for group_name, group_cols in feature_group_columns(cols).items():
        if not group_cols:
            rows.append(
                {
                    "feature_group": group_name,
                    "method": "skipped",
                    "num_features": 0,
                    "skip_reason": "no_features",
                }
            )
            continue
        for method, runner in [("logistic_regression", run_lr_protocol), ("photo_gmm", run_gmm_protocol)]:
            try:
                result = runner(train_df, val_df, test_df, group_cols, cfg, seed)
                summary = result["summary"]
                macro = summary.get("macro_average", {})
                worst = summary.get("worst_generator") or {}
                rows.append(
                    {
                        "feature_group": group_name,
                        "method": method,
                        "num_features": len(group_cols),
                        "threshold": finite_float(result["threshold"]),
                        "val_balanced_accuracy": finite_float(result["val_metrics"].get("balanced_accuracy")),
                        "macro_accuracy": finite_float(macro.get("accuracy")),
                        "macro_balanced_accuracy": finite_float(macro.get("balanced_accuracy")),
                        "macro_auroc": finite_float(macro.get("auroc")),
                        "macro_average_precision": finite_float(macro.get("average_precision")),
                        "worst_generator": worst.get("generator"),
                        "worst_balanced_accuracy": finite_float(worst.get("balanced_accuracy")),
                        "skip_reason": "",
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "feature_group": group_name,
                        "method": method,
                        "num_features": len(group_cols),
                        "skip_reason": str(exc),
                    }
                )
    return pd.DataFrame(rows)


def feature_group_for_column(col: str) -> str:
    if col.startswith("noise_"):
        return "noise"
    if col.startswith("hp_"):
        return "highpass"
    if col.startswith("channel_"):
        return "channel"
    if col.startswith("freq_"):
        return "frequency"
    if col.startswith("cfa_"):
        return "cfa_like"
    return "unknown"


def run_effect_size_analysis(train_df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    y_all = labels_from_df(train_df)
    for col in cols:
        values = train_df[col].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
        mask = np.isfinite(values)
        y = y_all[mask]
        vals = values[mask]
        real = vals[y == 0]
        fake = vals[y == 1]
        record: Dict[str, object] = {
            "feature": col,
            "feature_group": feature_group_for_column(col),
            "num_real": int(len(real)),
            "num_fake": int(len(fake)),
        }
        if len(real) < 2 or len(fake) < 2:
            record.update(
                {
                    "cohens_d_fake_minus_real": None,
                    "median_diff_fake_minus_real": None,
                    "auc_fake_higher": None,
                    "rank_biserial_fake_higher": None,
                    "abs_cohens_d": None,
                    "abs_rank_biserial": None,
                }
            )
        else:
            real_mean = float(np.mean(real))
            fake_mean = float(np.mean(fake))
            real_var = float(np.var(real, ddof=1))
            fake_var = float(np.var(fake, ddof=1))
            pooled = math.sqrt(((len(real) - 1) * real_var + (len(fake) - 1) * fake_var) / float(len(real) + len(fake) - 2))
            cohens_d = 0.0 if pooled == 0.0 else (fake_mean - real_mean) / pooled
            auc = roc_auc_score(y, vals) if len(np.unique(y)) > 1 else np.nan
            rank_biserial = 2.0 * auc - 1.0
            record.update(
                {
                    "cohens_d_fake_minus_real": finite_float(cohens_d),
                    "median_diff_fake_minus_real": finite_float(np.median(fake) - np.median(real)),
                    "auc_fake_higher": finite_float(auc),
                    "rank_biserial_fake_higher": finite_float(rank_biserial),
                    "abs_cohens_d": finite_float(abs(cohens_d)),
                    "abs_rank_biserial": finite_float(abs(rank_biserial)),
                }
            )
        records.append(record)
    df = pd.DataFrame(records)
    if not df.empty:
        df["_sort_key"] = df["abs_cohens_d"].fillna(0.0)
        df = df.sort_values(["_sort_key", "feature"], ascending=[False, True]).drop(columns=["_sort_key"])
    return df


def stratified_plot_sample(df: pd.DataFrame, max_rows: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df.copy()
    rng = np.random.RandomState(seed)
    buckets = []
    group_cols = [col for col in ["split", "generator", "label"] if col in df.columns]
    for _, group in df.groupby(group_cols, sort=True):
        buckets.append(group)
    per_bucket = max(1, int(math.ceil(float(max_rows) / float(max(1, len(buckets))))))
    sampled = []
    for group in buckets:
        if len(group) <= per_bucket:
            sampled.append(group)
        else:
            sampled.append(group.iloc[rng.choice(len(group), size=per_bucket, replace=False)])
    out = pd.concat(sampled, axis=0)
    if len(out) > max_rows:
        out = out.iloc[rng.choice(len(out), size=max_rows, replace=False)]
    return out.copy()


def embedding_coords(x_scaled: np.ndarray, seed: int, method: str) -> Tuple[np.ndarray, str]:
    if len(x_scaled) < 3:
        return PCA(n_components=2, random_state=seed).fit_transform(x_scaled), "PCA fallback"
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed).fit_transform(x_scaled), "UMAP"
        except Exception:
            return PCA(n_components=2, random_state=seed).fit_transform(x_scaled), "PCA fallback"
    perplexity = max(2, min(30, (len(x_scaled) - 1) // 3))
    return (
        TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate=200.0).fit_transform(
            x_scaled
        ),
        "t-SNE",
    )


def save_label_scatter(path: Path, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 6))
    for label, name, color in [(0, "real", "#2f6f9f"), (1, "fake", "#c44e52")]:
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=10, alpha=0.7, label=name, c=color)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_category_scatter(path: Path, coords: np.ndarray, categories: Sequence[str], title: str) -> None:
    ensure_dir(path.parent)
    categories = np.asarray(categories)
    names = sorted(pd.Series(categories).dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    plt.figure(figsize=(8, 6))
    for idx, name in enumerate(names):
        mask = categories == name
        plt.scatter(coords[mask, 0], coords[mask, 1], s=9, alpha=0.7, label=name, color=cmap(idx % 20))
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    if len(names) <= 30:
        plt.legend(frameon=False, fontsize=6, ncol=2, loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def save_embedding_plots(
    df: pd.DataFrame,
    train_df: pd.DataFrame,
    cols: Sequence[str],
    out_dir: Path,
    seed: int,
    max_rows: int,
) -> Dict[str, object]:
    plot_df = stratified_plot_sample(df, max_rows=max_rows, seed=seed).reset_index(drop=True)
    imputer = SimpleImputer(strategy="median").fit(matrix_from_df(train_df, cols))
    scaler = StandardScaler().fit(imputer.transform(matrix_from_df(train_df, cols)))
    x_plot = scaler.transform(imputer.transform(matrix_from_df(plot_df, cols)))
    outputs: Dict[str, object] = {"num_plot_rows": int(len(plot_df))}
    for method in ["umap", "tsne"]:
        coords, actual = embedding_coords(x_plot, seed=seed, method=method)
        outputs[f"{method}_method"] = actual
        save_label_scatter(
            out_dir / f"{method}_handcrafted_label.png",
            coords,
            labels_from_df(plot_df),
            "{} handcrafted features by label".format(actual),
        )
        save_category_scatter(
            out_dir / f"{method}_handcrafted_generator.png",
            coords,
            plot_df["generator"].astype(str).tolist(),
            "{} handcrafted features by generator".format(actual),
        )
    return outputs


def validate_official_splits(df: pd.DataFrame) -> None:
    required_cols = {"split", "generator", "generator_family", "label"}
    missing = required_cols.difference(df.columns)
    if missing:
        raise RuntimeError("Feature table is missing AIGIBench metadata columns: {}".format(sorted(missing)))
    for split in ["train", "val", "test"]:
        if split not in set(df["split"].astype(str)):
            raise RuntimeError("Feature table has no rows for split='{}'.".format(split))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    eval_cfg = cfg.get("aigibench_eval", {})
    out_dir = resolve_path(cfg, args.output_dir or eval_cfg.get("output_dir", "outputs/aigibench_handcrafted/evaluation"))
    ensure_dir(out_dir)
    expected = [
        out_dir / "probe_per_generator_metrics.csv",
        out_dir / "probe_summary.json",
        out_dir / "gmm_per_generator_metrics.csv",
        out_dir / "gmm_summary.json",
        out_dir / "feature_group_ablation.csv",
        out_dir / "feature_effect_size.csv",
        out_dir / "thresholds.json",
        out_dir / "confusion_by_generator_family.csv",
        out_dir / "umap_handcrafted_label.png",
        out_dir / "tsne_handcrafted_label.png",
    ]
    if all(path.exists() for path in expected) and not args.overwrite:
        print("AIGIBench handcrafted evaluation exists, skip: {}. Use --overwrite to rebuild.".format(out_dir))
        return

    features_path = resolve_path(cfg, args.features)
    df = load_table(features_path)
    validate_official_splits(df)
    cols = feature_columns(df)
    if not cols:
        raise RuntimeError("No numeric handcrafted RAW-like feature columns found.")
    df = df.copy()
    df["split"] = df["split"].astype(str)
    df["generator"] = df["generator"].astype(str)
    df["generator_family"] = df["generator_family"].astype(str)
    df["label"] = df["label"].astype(int)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    seed = int(eval_cfg.get("random_seed", cfg.get("project", {}).get("random_seed", 42)))

    lr = run_lr_protocol(train_df, val_df, test_df, cols, cfg, seed)
    gmm = run_gmm_protocol(train_df, val_df, test_df, cols, cfg, seed)

    write_dataframe_outputs(
        lr["per_generator"],
        out_dir / "probe_per_generator_metrics.csv",
        out_dir / "probe_per_generator_metrics.json",
    )
    write_json(
        out_dir / "probe_summary.json",
        {
            "method": "logistic_regression",
            "protocol": "SimpleImputer + StandardScaler fit on train; LogisticRegression fit on train; global threshold selected on pooled val; frozen evaluation on test.",
            "per_generator_composition": "Each test generator uses that generator subset's 0_real images as negatives and 1_fake images as positives.",
            "summary": lr["summary"],
            "val_metrics_at_selected_threshold": lr["val_metrics"],
            "smoke_run": bool(args.smoke),
            "smoke_note": "Smoke metrics validate only pipeline execution and must not be used as experimental conclusions." if args.smoke else "",
        },
    )

    write_dataframe_outputs(
        gmm["per_generator"],
        out_dir / "gmm_per_generator_metrics.csv",
        out_dir / "gmm_per_generator_metrics.json",
    )
    gmm_protocol = gmm["protocol"]
    write_json(
        out_dir / "gmm_summary.json",
        {
            "method": "photo_gmm",
            "protocol": "SimpleImputer + StandardScaler + PCA + GaussianMixture fit only on train real/photo features; global anomaly threshold selected on pooled val; frozen evaluation on test.",
            "score": "anomaly_score = -log_likelihood",
            "pca_components_fit": int(gmm_protocol.pca_components_fit),
            "gmm_components_fit": int(gmm_protocol.n_components_fit),
            "reg_covar": float(gmm_protocol.reg_covar),
            "per_generator_composition": "Each test generator uses that generator subset's 0_real images as negatives and 1_fake images as positives.",
            "summary": gmm["summary"],
            "val_metrics_at_selected_threshold": gmm["val_metrics"],
            "smoke_run": bool(args.smoke),
            "smoke_note": "Smoke metrics validate only pipeline execution and must not be used as experimental conclusions." if args.smoke else "",
        },
    )
    save_gmm_likelihood_hist(test_df, gmm["test_scores"], out_dir / "gmm_likelihood_hist.png")

    thresholds = {
        "logistic_regression": {
            "threshold": finite_float(lr["threshold"]),
            "selected_on": "pooled validation split",
            "selection_metric": "balanced_accuracy",
            "val_metrics": lr["val_metrics"],
        },
        "photo_gmm": {
            "threshold": finite_float(gmm["threshold"]),
            "selected_on": "pooled validation split",
            "selection_metric": "balanced_accuracy",
            "score": "anomaly_score = -log_likelihood",
            "val_metrics": gmm["val_metrics"],
        },
    }
    write_json(out_dir / "thresholds.json", thresholds)

    family_rows = []
    family_rows.extend(confusion_by_family(test_df, lr["test_scores"], lr["threshold"], "logistic_regression"))
    family_rows.extend(confusion_by_family(test_df, gmm["test_scores"], gmm["threshold"], "photo_gmm"))
    family_df = pd.DataFrame(family_rows)
    write_dataframe_outputs(
        family_df,
        out_dir / "confusion_by_generator_family.csv",
        out_dir / "confusion_by_generator_family.json",
    )

    ablation_df = run_feature_group_ablation(train_df, val_df, test_df, cols, cfg, seed)
    write_dataframe_outputs(ablation_df, out_dir / "feature_group_ablation.csv", out_dir / "feature_group_ablation.json")

    effect_df = run_effect_size_analysis(train_df, cols)
    write_dataframe_outputs(effect_df, out_dir / "feature_effect_size.csv", out_dir / "feature_effect_size.json")
    top_k = int(eval_cfg.get("effect_size_top_k", 50))
    write_json(
        out_dir / "feature_effect_size_top.json",
        {"top_by_abs_cohens_d": effect_df.head(top_k).where(pd.notnull(effect_df.head(top_k)), None).to_dict(orient="records")},
    )

    plot_info = save_embedding_plots(
        df,
        train_df=train_df,
        cols=cols,
        out_dir=out_dir,
        seed=seed,
        max_rows=int(eval_cfg.get("plot_sample_size", 10000)),
    )
    write_json(
        out_dir / "evaluation_summary.json",
        {
            "features_path": str(features_path),
            "num_rows": int(len(df)),
            "num_features": int(len(cols)),
            "split_counts": {str(k): int(v) for k, v in df["split"].value_counts().sort_index().items()},
            "label_counts_by_split": {
                "{}:{}".format(split, label): int(count)
                for (split, label), count in df.groupby(["split", "label"]).size().sort_index().items()
            },
            "plot_info": plot_info,
            "smoke_run": bool(args.smoke),
            "smoke_note": "Smoke metrics validate only pipeline execution and must not be used as experimental conclusions." if args.smoke else "",
        },
    )

    copy_config(cfg, resolve_path(cfg, "outputs"))
    update_run_metadata(
        cfg,
        aigibench_handcrafted_eval_output_dir=str(out_dir),
        aigibench_handcrafted_num_features=len(cols),
        aigibench_handcrafted_lr_macro=lr["summary"].get("macro_average"),
        aigibench_handcrafted_gmm_macro=gmm["summary"].get("macro_average"),
    )
    print("Wrote AIGIBench handcrafted evaluation under: {}".format(out_dir))


if __name__ == "__main__":
    main()
