#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import ensure_dir, load_config, set_seed, write_json
from src.features.utils import feature_group_columns
from src.io_utils import stable_feature_columns
from src.training.metrics import (
    binary_metrics,
    confusion_by_generator_family,
    per_generator_metrics,
    select_global_threshold,
    summarize_per_generator,
)


def read_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def split_xy(df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    return (
        train,
        val,
        test,
        train[feature_columns].replace([np.inf, -np.inf], np.nan).values.astype(np.float32),
        val[feature_columns].replace([np.inf, -np.inf], np.nan).values.astype(np.float32),
        test[feature_columns].replace([np.inf, -np.inf], np.nan).values.astype(np.float32),
    )


def classifier_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1].astype(float)
    scores = model.decision_function(x).astype(float)
    return scores


def eval_classifier_baseline(
    name: str,
    estimator: Any,
    df: pd.DataFrame,
    feature_columns: List[str],
    threshold_metric: str,
) -> Dict[str, Any]:
    train, val, test, x_train, x_val, x_test = split_xy(df, feature_columns)
    y_train = train["label"].astype(int).values
    y_val = val["label"].astype(int).values
    y_test = test["label"].astype(int).values
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )
    pipe.fit(x_train, y_train)
    val_scores = classifier_scores(pipe, x_val)
    test_scores = classifier_scores(pipe, x_test)
    threshold, threshold_info = select_global_threshold(y_val, val_scores, metric=threshold_metric)
    per_gen = per_generator_metrics(test, test_scores, threshold, split="test")
    return {
        "name": name,
        "threshold": float(threshold),
        "threshold_info": threshold_info,
        "val": binary_metrics(y_val, val_scores, threshold),
        "test": binary_metrics(y_test, test_scores, threshold),
        "per_generator_summary": summarize_per_generator(per_gen),
        "per_generator": per_gen,
        "family_confusion": confusion_by_generator_family(per_gen),
    }


def eval_gmm_baseline(
    df: pd.DataFrame,
    feature_columns: List[str],
    gmm_cfg: Dict[str, Any],
    threshold_metric: str,
) -> Dict[str, Any]:
    train, val, test, x_train, x_val, x_test = split_xy(df, feature_columns)
    y_train = train["label"].astype(int).values
    y_val = val["label"].astype(int).values
    y_test = test["label"].astype(int).values
    x_train_real = x_train[y_train == 0]
    if len(x_train_real) < 2:
        raise ValueError("GMM requires at least two train-real samples")
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    x_train_real_imp = imputer.fit_transform(x_train_real)
    x_train_real_scaled = scaler.fit_transform(x_train_real_imp)
    n_components = int(gmm_cfg.get("n_components", 4))
    pca_components = int(gmm_cfg.get("pca_components", 32))
    n_pca = max(1, min(pca_components, x_train_real_scaled.shape[0] - 1, x_train_real_scaled.shape[1]))
    pca = PCA(n_components=n_pca, random_state=int(gmm_cfg.get("random_state", 42)))
    x_train_real_pca = pca.fit_transform(x_train_real_scaled)
    gmm = GaussianMixture(
        n_components=max(1, min(n_components, len(x_train_real_pca))),
        covariance_type="full",
        reg_covar=float(gmm_cfg.get("reg_covar", 1e-5)),
        random_state=int(gmm_cfg.get("random_state", 42)),
    )
    gmm.fit(x_train_real_pca)

    def score(x: np.ndarray) -> np.ndarray:
        z = imputer.transform(x)
        z = scaler.transform(z)
        z = pca.transform(z)
        return -gmm.score_samples(z)

    val_scores = score(x_val)
    test_scores = score(x_test)
    threshold, threshold_info = select_global_threshold(y_val, val_scores, metric=threshold_metric)
    per_gen = per_generator_metrics(test, test_scores, threshold, split="test")
    return {
        "name": "gmm_real_likelihood",
        "threshold": float(threshold),
        "threshold_info": threshold_info,
        "pca_components": int(n_pca),
        "gmm_components": int(gmm.n_components),
        "val": binary_metrics(y_val, val_scores, threshold),
        "test": binary_metrics(y_test, test_scores, threshold),
        "per_generator_summary": summarize_per_generator(per_gen),
        "per_generator": per_gen,
        "family_confusion": confusion_by_generator_family(per_gen),
    }


def feature_effect_size(df: pd.DataFrame, feature_columns: List[str], top_k: int) -> pd.DataFrame:
    train = df[df["split"] == "train"].copy()
    y = train["label"].astype(int).values
    rows = []
    for col in feature_columns:
        x = pd.to_numeric(train[col], errors="coerce").replace([np.inf, -np.inf], np.nan).values.astype(float)
        median = np.nanmedian(x)
        x = np.where(np.isfinite(x), x, median)
        real = x[y == 0]
        fake = x[y == 1]
        if len(real) == 0 or len(fake) == 0:
            continue
        mean_real, mean_fake = float(np.mean(real)), float(np.mean(fake))
        var_real, var_fake = float(np.var(real, ddof=1)), float(np.var(fake, ddof=1))
        pooled = np.sqrt(0.5 * (var_real + var_fake) + 1e-12)
        cohen_d = (mean_fake - mean_real) / pooled
        try:
            auc = float(roc_auc_score(y, x))
        except Exception:
            auc = float("nan")
        rows.append(
            {
                "feature": col,
                "group": col.split("_", 1)[0],
                "mean_real": mean_real,
                "mean_fake": mean_fake,
                "cohen_d_fake_minus_real": float(cohen_d),
                "abs_cohen_d": float(abs(cohen_d)),
                "median_diff_fake_minus_real": float(np.median(fake) - np.median(real)),
                "auc_effect": auc,
                "abs_auc_effect_from_0_5": float(abs(auc - 0.5)) if np.isfinite(auc) else float("nan"),
            }
        )
    out = pd.DataFrame(rows).sort_values(["abs_cohen_d", "abs_auc_effect_from_0_5"], ascending=False)
    return out.head(int(top_k)).reset_index(drop=True)


def _fit_visual_embedding(x: np.ndarray, method: str, random_state: int, perplexity: int) -> np.ndarray:
    x = SimpleImputer(strategy="median").fit_transform(x)
    x = StandardScaler().fit_transform(x)
    n_pca = min(50, x.shape[1], max(1, x.shape[0] - 1))
    x = PCA(n_components=n_pca, random_state=random_state).fit_transform(x)
    if method == "tsne":
        from sklearn.manifold import TSNE

        pp = max(5, min(int(perplexity), max(5, (len(x) - 1) // 3)))
        return TSNE(n_components=2, perplexity=pp, init="pca", learning_rate=200.0, random_state=random_state).fit_transform(x)
    if method == "umap":
        import umap

        return umap.UMAP(n_components=2, random_state=random_state).fit_transform(x)
    raise KeyError(method)


def make_plots(df: pd.DataFrame, feature_columns: List[str], analysis_dir: str, cfg: Dict[str, Any]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plots: {exc}")
        return
    plots_dir = ensure_dir(os.path.join(analysis_dir, "plots"))
    sample_n = int(cfg.get("plot_sample", 500))
    random_state = int(cfg.get("random_state", 42))
    sample = df.sample(n=min(sample_n, len(df)), random_state=random_state).copy()

    def plot_embedding(emb: np.ndarray, color_col: str, title: str, path: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 5))
        values = sample[color_col].astype(str).values if color_col in sample.columns else sample["label"].astype(str).values
        uniques = sorted(set(values.tolist()))
        cmap = plt.get_cmap("tab20")
        for idx, value in enumerate(uniques):
            mask = values == value
            ax.scatter(emb[mask, 0], emb[mask, 1], s=10, alpha=0.75, label=value, color=cmap(idx % 20))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        if len(uniques) <= 15:
            ax.legend(fontsize=7, markerscale=1.5, frameon=False)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)

    plot_groups = ["all", "noise", "srm", "glcm", "frequency"]
    for group in plot_groups:
        cols = feature_group_columns(feature_columns, group)
        if not cols:
            continue
        x = sample[cols].values.astype(np.float32)
        for method, enabled in [("umap", bool(cfg.get("run_umap", True))), ("tsne", bool(cfg.get("run_tsne", True)))]:
            if not enabled:
                continue
            try:
                emb = _fit_visual_embedding(x, method, random_state, int(cfg.get("tsne_perplexity", 30)))
            except Exception as exc:
                print(f"Skip {method} plot for {group}: {exc}")
                continue
            prefix = "handcrafted" if group == "all" else group
            plot_embedding(emb, "label", f"{method.upper()} {group} by label", os.path.join(plots_dir, f"{method}_{prefix}_label.png"))
            color_col = "generator" if "generator" in sample.columns else "subset"
            plot_embedding(emb, color_col, f"{method.upper()} {group} by generator", os.path.join(plots_dir, f"{method}_{prefix}_generator.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    set_seed(int(config.get("project", {}).get("seed", 42)))
    out_root = config["paths"]["output_dir"]
    analysis_dir = ensure_dir(os.path.join(out_root, "analysis"))
    features_path = args.features or os.path.join(out_root, "features.parquet")
    df = read_features(features_path)
    feature_columns = stable_feature_columns(df)
    threshold_metric = str(config.get("mlp", {}).get("threshold_metric", "balanced_accuracy"))
    seed = int(config.get("project", {}).get("seed", 42))

    baseline_specs = [
        ("logistic_regression", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed)),
        ("linear_svm", LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000)),
        ("random_forest", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)),
    ]
    baseline_results: Dict[str, Any] = {}
    for name, estimator in baseline_specs:
        result = eval_classifier_baseline(name, estimator, df, feature_columns, threshold_metric)
        per_gen = result.pop("per_generator")
        family = result.pop("family_confusion")
        per_gen.to_csv(os.path.join(analysis_dir, f"{name}_per_generator_metrics.csv"), index=False)
        if not family.empty:
            family.to_csv(os.path.join(analysis_dir, f"{name}_confusion_by_generator_family.csv"), index=False)
        baseline_results[name] = result

    gmm_cfg = dict(config.get("baselines", {}).get("gmm", {}))
    gmm_cfg["random_state"] = seed
    gmm_result = eval_gmm_baseline(df, feature_columns, gmm_cfg, threshold_metric)
    gmm_per_gen = gmm_result.pop("per_generator")
    gmm_family = gmm_result.pop("family_confusion")
    gmm_per_gen.to_csv(os.path.join(analysis_dir, "gmm_per_generator_metrics.csv"), index=False)
    gmm_per_gen.to_json(os.path.join(analysis_dir, "gmm_per_generator_metrics.json"), orient="records", indent=2)
    if not gmm_family.empty:
        gmm_family.to_csv(os.path.join(analysis_dir, "gmm_confusion_by_generator_family.csv"), index=False)
    baseline_results["gmm_real_likelihood"] = gmm_result
    write_json(baseline_results, os.path.join(analysis_dir, "baseline_metrics.json"))

    effect = feature_effect_size(df, feature_columns, int(config.get("analysis", {}).get("effect_size_top_k", 100)))
    effect.to_csv(os.path.join(analysis_dir, "feature_effect_size.csv"), index=False)
    effect.to_json(os.path.join(analysis_dir, "feature_effect_size.json"), orient="records", indent=2)
    make_plots(df, feature_columns, analysis_dir, config.get("analysis", {}))
    print(f"Wrote analysis under: {analysis_dir}")


if __name__ == "__main__":
    main()
