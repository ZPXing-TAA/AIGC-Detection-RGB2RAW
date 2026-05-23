#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import ensure_dir, load_config, set_seed, write_json

EPS = 1e-12


@dataclass
class FittedTransform:
    feature_columns: List[str]
    imputer: SimpleImputer
    scaler: StandardScaler
    pca: Optional[PCA]

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_columns].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
        z = self.imputer.transform(x)
        z = self.scaler.transform(z)
        if self.pca is not None:
            z = self.pca.transform(z)
        return z.astype(np.float64)


@dataclass
class ScoredModel:
    feature_set: str
    model_name: str
    model_params: Dict[str, Any]
    pca_dim: Optional[int]
    transform: FittedTransform
    model: Any
    score_train_real: np.ndarray
    score_val_real: np.ndarray
    score_test: np.ndarray
    score_val_all: np.ndarray


def read_features(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def feature_columns_for_set(df: pd.DataFrame, feature_set: str, config: Dict[str, Any]) -> List[str]:
    spec = config["feature_sets"][feature_set]
    prefixes = tuple(spec.get("prefixes", []))
    cols = [c for c in df.columns if c.startswith(prefixes)]
    if not cols:
        raise ValueError(f"feature set {feature_set} has no columns for prefixes {prefixes}")
    return cols


def allowed_pca_dim(requested: Optional[int], n_samples: int, n_features: int) -> Optional[int]:
    if requested is None:
        return None
    dim = min(int(requested), max(1, n_samples - 1), n_features)
    return dim if dim >= 1 else None


def fit_transformer(train_real: pd.DataFrame, feature_columns: List[str], pca_dim: Optional[int], seed: int) -> FittedTransform:
    x = train_real[feature_columns].replace([np.inf, -np.inf], np.nan).values.astype(np.float32)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    z = imputer.fit_transform(x)
    z = scaler.fit_transform(z)
    pca = None
    dim = allowed_pca_dim(pca_dim, z.shape[0], z.shape[1])
    if dim is not None:
        pca = PCA(n_components=dim, random_state=seed)
        pca.fit(z)
    return FittedTransform(feature_columns=list(feature_columns), imputer=imputer, scaler=scaler, pca=pca)


def safe_auroc(y: np.ndarray, score: np.ndarray) -> float:
    return float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def safe_aupr(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if len(np.unique(y)) == 2 else float("nan")


def fit_diagonal_gaussian(z_train: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "mean": np.mean(z_train, axis=0),
        "var": np.var(z_train, axis=0) + EPS,
    }


def score_diagonal_gaussian(model: Dict[str, np.ndarray], z: np.ndarray) -> np.ndarray:
    return np.sum(((z - model["mean"]) ** 2) / model["var"], axis=1)


def fit_full_gaussian(z_train: np.ndarray, reg_lambda: float) -> Dict[str, np.ndarray]:
    mean = np.mean(z_train, axis=0)
    centered = z_train - mean
    cov = np.cov(centered, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + float(reg_lambda) * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov)
    return {"mean": mean, "inv_cov": inv}


def score_full_gaussian(model: Dict[str, np.ndarray], z: np.ndarray) -> np.ndarray:
    centered = z - model["mean"]
    return np.einsum("ij,jk,ik->i", centered, model["inv_cov"], centered)


def score_model(model_name: str, model: Any, z: np.ndarray) -> np.ndarray:
    if model_name == "diagonal_gaussian":
        return score_diagonal_gaussian(model, z)
    if model_name == "full_gaussian":
        return score_full_gaussian(model, z)
    if model_name == "pca_gmm":
        return -model.score_samples(z)
    if model_name == "knn":
        distances, _ = model.kneighbors(z, return_distance=True)
        return np.mean(distances, axis=1)
    if model_name == "isolation_forest":
        return -model.decision_function(z)
    if model_name == "one_class_svm":
        return -model.decision_function(z)
    raise KeyError(model_name)


def build_model_specs(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid = config["photo_only"]["model_grid"]
    pca_dims = list(config["photo_only"].get("pca_dims", [None, 8, 16, 32]))
    specs: List[Dict[str, Any]] = []
    if grid.get("diagonal_gaussian", {}).get("enabled", True):
        for pca_dim in pca_dims:
            specs.append({"model": "diagonal_gaussian", "pca_dim": pca_dim, "params": {}})
    if grid.get("full_gaussian", {}).get("enabled", True):
        for pca_dim in pca_dims:
            for reg in grid["full_gaussian"].get("reg_lambdas", [1e-4]):
                specs.append({"model": "full_gaussian", "pca_dim": pca_dim, "params": {"reg_lambda": float(reg)}})
    if grid.get("pca_gmm", {}).get("enabled", True):
        for pca_dim in grid["pca_gmm"].get("pca_dims", [8, 16, 32]):
            for k in grid["pca_gmm"].get("n_components", [2, 4]):
                for reg in grid["pca_gmm"].get("reg_covar", [1e-5]):
                    specs.append({"model": "pca_gmm", "pca_dim": pca_dim, "params": {"n_components": int(k), "reg_covar": float(reg)}})
    if grid.get("knn", {}).get("enabled", True):
        for pca_dim in pca_dims:
            for k in grid["knn"].get("k", [5, 10]):
                specs.append({"model": "knn", "pca_dim": pca_dim, "params": {"k": int(k)}})
    if grid.get("isolation_forest", {}).get("enabled", True):
        for pca_dim in [None, 32]:
            for n in grid["isolation_forest"].get("n_estimators", [200]):
                specs.append({"model": "isolation_forest", "pca_dim": pca_dim, "params": {"n_estimators": int(n)}})
    if grid.get("one_class_svm", {}).get("enabled", True):
        for pca_dim in grid["one_class_svm"].get("pca_dims", [8, 16, 32]):
            for nu in grid["one_class_svm"].get("nu", [0.05]):
                specs.append({"model": "one_class_svm", "pca_dim": pca_dim, "params": {"nu": float(nu)}})
    return specs


def fit_one_model(
    feature_set: str,
    spec: Dict[str, Any],
    train_real: pd.DataFrame,
    val_real: pd.DataFrame,
    val_all: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: List[str],
    seed: int,
) -> ScoredModel:
    transformer = fit_transformer(train_real, feature_columns, spec["pca_dim"], seed)
    z_train = transformer.transform(train_real)
    z_val_real = transformer.transform(val_real)
    z_val_all = transformer.transform(val_all)
    z_test = transformer.transform(test)
    name = spec["model"]
    params = dict(spec["params"])

    if name == "diagonal_gaussian":
        model = fit_diagonal_gaussian(z_train)
    elif name == "full_gaussian":
        model = fit_full_gaussian(z_train, params["reg_lambda"])
    elif name == "pca_gmm":
        k = min(int(params["n_components"]), len(z_train))
        model = GaussianMixture(
            n_components=max(1, k),
            covariance_type="full",
            reg_covar=float(params["reg_covar"]),
            random_state=seed,
        )
        model.fit(z_train)
        params["n_components_effective"] = int(model.n_components)
    elif name == "knn":
        k = min(int(params["k"]), len(z_train))
        model = NearestNeighbors(n_neighbors=max(1, k), metric="euclidean")
        model.fit(z_train)
        params["k_effective"] = int(max(1, k))
    elif name == "isolation_forest":
        model = IsolationForest(
            n_estimators=int(params["n_estimators"]),
            max_samples="auto",
            contamination="auto",
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(z_train)
    elif name == "one_class_svm":
        model = OneClassSVM(kernel="rbf", nu=float(params["nu"]), gamma="scale")
        model.fit(z_train)
    else:
        raise KeyError(name)

    return ScoredModel(
        feature_set=feature_set,
        model_name=name,
        model_params=params,
        pca_dim=transformer.pca.n_components_ if transformer.pca is not None else None,
        transform=transformer,
        model=model,
        score_train_real=score_model(name, model, z_train),
        score_val_real=score_model(name, model, z_val_real),
        score_val_all=score_model(name, model, z_val_all),
        score_test=score_model(name, model, z_test),
    )


def threshold_from_val_real(scores: np.ndarray, alpha: float) -> float:
    return float(np.quantile(scores, 1.0 - float(alpha)))


def fixed_fpr_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    pred_anom = score > float(threshold)
    real = y == 0
    fake = y == 1
    real_fpr = float(np.mean(pred_anom[real])) if np.any(real) else float("nan")
    fake_tpr = float(np.mean(pred_anom[fake])) if np.any(fake) else float("nan")
    photo_acc = 1.0 - real_fpr if np.isfinite(real_fpr) else float("nan")
    gen_acc = fake_tpr
    overall = float(np.mean(pred_anom == fake)) if len(y) else float("nan")
    return {
        "real_fpr": real_fpr,
        "fake_tpr": fake_tpr,
        "photographic_acc": photo_acc,
        "generated_acc": gen_acc,
        "overall_acc": overall,
    }


def evaluate_scored_model(
    scored: ScoredModel,
    train_real: pd.DataFrame,
    val_real: pd.DataFrame,
    val_all: pd.DataFrame,
    test: pd.DataFrame,
    threshold_fprs: Iterable[float],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    y_test = test["label"].astype(int).values
    y_val_all = val_all["label"].astype(int).values
    test_auroc = safe_auroc(y_test, scored.score_test)
    test_aupr = safe_aupr(y_test, scored.score_test)
    val_diag_auroc = safe_auroc(y_val_all, scored.score_val_all)
    val_diag_aupr = safe_aupr(y_val_all, scored.score_val_all)
    threshold_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    per_gen_rows: List[Dict[str, Any]] = []

    for alpha in threshold_fprs:
        tau = threshold_from_val_real(scored.score_val_real, float(alpha))
        train_real_fpr = float(np.mean(scored.score_train_real > tau))
        val_real_fpr = float(np.mean(scored.score_val_real > tau))
        m = fixed_fpr_metrics(y_test, scored.score_test, tau)
        gen_rows = per_generator_rows(scored, test, tau, float(alpha))
        valid_gen = [r for r in gen_rows if np.isfinite(r["fake_tpr"]) and np.isfinite(r["AUROC"])]
        macro_tpr = float(np.mean([r["fake_tpr"] for r in valid_gen])) if valid_gen else float("nan")
        worst = min(valid_gen, key=lambda r: r["fake_tpr"]) if valid_gen else {}
        threshold_rows.append(
            {
                "feature_set": scored.feature_set,
                "model": scored.model_name,
                "model_params": scored.model_params,
                "pca_dim": scored.pca_dim,
                "threshold_fpr": float(alpha),
                "threshold": tau,
                "train_real_fpr": train_real_fpr,
                "val_real_fpr": val_real_fpr,
            }
        )
        summary_rows.append(
            {
                "feature_set": scored.feature_set,
                "n_features": len(scored.transform.feature_columns),
                "model": scored.model_name,
                "model_params": scored.model_params,
                "pca_dim": scored.pca_dim,
                "threshold_fpr": float(alpha),
                "val_real_fpr": val_real_fpr,
                "test_real_fpr": m["real_fpr"],
                "test_fake_tpr": m["fake_tpr"],
                "test_photographic_acc": m["photographic_acc"],
                "test_generated_acc": m["generated_acc"],
                "test_overall_acc": m["overall_acc"],
                "test_AUROC": test_auroc,
                "test_AUPR": test_aupr,
                "macro_generator_tpr": macro_tpr,
                "worst_generator": worst.get("generator", ""),
                "worst_generator_tpr": worst.get("fake_tpr", float("nan")),
                "diagnostic_val_AUROC_not_photo_only": val_diag_auroc,
                "diagnostic_val_AUPR_not_photo_only": val_diag_aupr,
            }
        )
        per_gen_rows.extend(gen_rows)
    return summary_rows, per_gen_rows, threshold_rows


def per_generator_rows(scored: ScoredModel, test: pd.DataFrame, threshold: float, alpha: float) -> List[Dict[str, Any]]:
    work = test.copy()
    work["score"] = scored.score_test
    rows = []
    group_col = "subset" if "subset" in work.columns else "generator"
    for subset, part in work.groupby(group_col, sort=True):
        y = part["label"].astype(int).values
        score = part["score"].values.astype(float)
        m = fixed_fpr_metrics(y, score, threshold)
        rows.append(
            {
                "feature_set": scored.feature_set,
                "model": scored.model_name,
                "model_params": scored.model_params,
                "pca_dim": scored.pca_dim,
                "threshold_fpr": float(alpha),
                "generator": str(part["generator"].iloc[0]) if "generator" in part.columns and len(part) else str(subset),
                "generator_family": str(part["generator_family"].iloc[0]) if "generator_family" in part.columns and len(part) else "",
                "n_real": int(np.sum(y == 0)),
                "n_fake": int(np.sum(y == 1)),
                "real_fpr": m["real_fpr"],
                "fake_tpr": m["fake_tpr"],
                "photographic_acc": m["photographic_acc"],
                "generated_acc": m["generated_acc"],
                "overall_acc": m["overall_acc"],
                "AUROC": safe_auroc(y, score),
                "AUPR": safe_aupr(y, score),
            }
        )
    return rows


def model_id(row: Dict[str, Any]) -> str:
    params = row.get("model_params", {})
    pieces = [str(row["feature_set"]), str(row["model"]), f"pca{row.get('pca_dim')}"]
    for k in sorted(params):
        pieces.append(f"{k}{params[k]}")
    return "__".join(pieces).replace("/", "_")


def make_plots(
    df: pd.DataFrame,
    scored: ScoredModel,
    train_real: pd.DataFrame,
    val_real: pd.DataFrame,
    test: pd.DataFrame,
    plots_dir: str,
    alpha: float,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skip plots: {exc}")
        return

    ensure_dir(plots_dir)
    tau = threshold_from_val_real(scored.score_val_real, alpha)
    test_real = test[test["label"] == 0].copy()
    test_fake = test[test["label"] == 1].copy()
    score_test_real = scored.score_test[test["label"].astype(int).values == 0]
    score_test_fake = scored.score_test[test["label"].astype(int).values == 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(scored.score_train_real, bins=60, alpha=0.45, density=True, label="train real")
    ax.hist(scored.score_val_real, bins=60, alpha=0.45, density=True, label="val real")
    ax.hist(score_test_real, bins=60, alpha=0.45, density=True, label="test real")
    ax.hist(score_test_fake, bins=60, alpha=0.45, density=True, label="test fake")
    ax.axvline(tau, color="black", linestyle="--", linewidth=1, label=f"val real {alpha:.0%} FPR")
    ax.set_title(f"Score histogram: {scored.feature_set} / {scored.model_name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "score_histogram_main.png"), dpi=180)
    plt.close(fig)

    y = test["label"].astype(int).values
    fpr, tpr, _ = roc_curve(y, scored.score_test)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    ax.set_xlabel("Real FPR")
    ax.set_ylabel("Fake TPR")
    ax.set_title("Test ROC")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "roc_curve.png"), dpi=180)
    plt.close(fig)

    gen_rows = per_generator_rows(scored, test, tau, alpha)
    gen_df = pd.DataFrame(gen_rows).sort_values("fake_tpr")
    fig, ax = plt.subplots(figsize=(max(8, len(gen_df) * 0.35), 4))
    ax.bar(gen_df["generator"], gen_df["fake_tpr"])
    ax.set_ylabel("TPR")
    ax.set_title(f"Per-generator TPR at {alpha:.0%} real FPR")
    ax.tick_params(axis="x", rotation=75, labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "tpr_at_fpr_by_generator.png"), dpi=180)
    plt.close(fig)

    # Diagnostic-only PCA 2D visualization. It uses fake labels only for coloring.
    sample = test.sample(n=min(2000, len(test)), random_state=42)
    z = scored.transform.transform(sample)
    if z.shape[1] > 2:
        z2 = PCA(n_components=2, random_state=42).fit_transform(z)
    else:
        z2 = z[:, :2]
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = sample["label"].astype(int).values
    ax.scatter(z2[colors == 0, 0], z2[colors == 0, 1], s=8, alpha=0.55, label="real")
    ax.scatter(z2[colors == 1, 0], z2[colors == 1, 1], s=8, alpha=0.55, label="fake")
    ax.set_title("PCA 2D diagnostic only")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "pca_2d_diagnostic_only.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--features", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--feature-set", action="append", default=None)
    parser.add_argument("--max-models", type=int, default=None)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    seed = int(config.get("project", {}).get("seed", 42))
    set_seed(seed)
    out_root = config["paths"]["output_dir"]
    features_path = args.features or os.path.join(out_root, "features.parquet")
    analysis_dir = ensure_dir(os.path.join(out_root, "photo_only_anomaly"))
    models_dir = ensure_dir(os.path.join(analysis_dir, "models"))
    plots_dir = ensure_dir(os.path.join(analysis_dir, "plots"))
    df = read_features(features_path)

    train_real = df[(df["split"] == "train") & (df["label"].astype(int) == 0)].copy()
    val_real = df[(df["split"] == "val") & (df["label"].astype(int) == 0)].copy()
    val_all = df[df["split"] == "val"].copy()
    test = df[df["split"] == "test"].copy()
    if train_real.empty or val_real.empty or test.empty:
        raise ValueError("requires train-real, val-real, and test rows")

    feature_sets = args.feature_set or list(config["feature_sets"].keys())
    specs = build_model_specs(config)
    if args.max_models is not None:
        specs = specs[: int(args.max_models)]
    threshold_fprs = [float(a) for a in config["photo_only"].get("threshold_fprs", [0.01, 0.05, 0.10])]

    summary_rows: List[Dict[str, Any]] = []
    per_gen_rows: List[Dict[str, Any]] = []
    threshold_rows: List[Dict[str, Any]] = []
    registry: List[Dict[str, Any]] = []
    best_for_plot: Optional[Tuple[ScoredModel, float]] = None
    best_plot_auc = -np.inf

    for feature_set in feature_sets:
        cols = feature_columns_for_set(df, feature_set, config)
        for spec in specs:
            try:
                scored = fit_one_model(feature_set, spec, train_real, val_real, val_all, test, cols, seed)
                model_summary, model_per_gen, model_thresholds = evaluate_scored_model(
                    scored, train_real, val_real, val_all, test, threshold_fprs
                )
                summary_rows.extend(model_summary)
                per_gen_rows.extend(model_per_gen)
                threshold_rows.extend(model_thresholds)
                this_auc = model_summary[0]["test_AUROC"] if model_summary else float("nan")
                if np.isfinite(this_auc) and this_auc > best_plot_auc:
                    best_plot_auc = float(this_auc)
                    best_for_plot = (scored, 0.05)
                entry = {
                    "model_id": model_id({"feature_set": feature_set, "model": scored.model_name, "pca_dim": scored.pca_dim, "model_params": scored.model_params}),
                    "feature_set": feature_set,
                    "n_features": len(cols),
                    "model": scored.model_name,
                    "model_params": scored.model_params,
                    "pca_dim": scored.pca_dim,
                    "preprocessing_fit": "train-real only",
                    "threshold_fit": "val-real only",
                    "fake_used_for_fit_or_threshold": False,
                }
                registry.append(entry)
                joblib.dump(
                    {
                        "transform": scored.transform,
                        "model_name": scored.model_name,
                        "model_params": scored.model_params,
                        "model": scored.model,
                    },
                    os.path.join(models_dir, entry["model_id"] + ".joblib"),
                )
            except Exception as exc:
                registry.append(
                    {
                        "feature_set": feature_set,
                        "model": spec["model"],
                        "model_params": spec["params"],
                        "pca_dim_requested": spec["pca_dim"],
                        "failed": True,
                        "error": repr(exc),
                    }
                )

    summary = pd.DataFrame(summary_rows)
    per_gen = pd.DataFrame(per_gen_rows)
    thresholds = pd.DataFrame(threshold_rows)
    summary.to_csv(os.path.join(analysis_dir, "main_summary.csv"), index=False)
    summary.to_json(os.path.join(analysis_dir, "main_summary.json"), orient="records", indent=2)
    per_gen.to_csv(os.path.join(analysis_dir, "per_generator_metrics.csv"), index=False)
    per_gen.to_json(os.path.join(analysis_dir, "per_generator_metrics.json"), orient="records", indent=2)
    thresholds.to_csv(os.path.join(analysis_dir, "thresholds.csv"), index=False)
    thresholds.to_json(os.path.join(analysis_dir, "thresholds.json"), orient="records", indent=2)
    write_json(registry, os.path.join(analysis_dir, "model_registry.json"))

    if not summary.empty:
        strict = (
            summary[summary["threshold_fpr"] == 0.05]
            .sort_values(["feature_set", "model", "pca_dim"], na_position="first")
            .groupby("feature_set", as_index=False)
            .head(1)
        )
        diagnostic = summary[summary["threshold_fpr"] == 0.05].sort_values("test_AUROC", ascending=False).head(20)
        strict.to_csv(os.path.join(analysis_dir, "strict_photo_only_selection.csv"), index=False)
        diagnostic.to_csv(os.path.join(analysis_dir, "diagnostic_selection_not_photo_only.csv"), index=False)
    if bool(config.get("analysis", {}).get("make_plots", True)) and not args.no_plots and best_for_plot is not None:
        make_plots(df, best_for_plot[0], train_real, val_real, test, plots_dir, best_for_plot[1])

    write_json(
        {
            "protocol": "photo-only anomaly detection",
            "train_fit_rows": int(len(train_real)),
            "threshold_calibration_rows": int(len(val_real)),
            "test_rows": int(len(test)),
            "fake_used_for_fit_or_threshold": False,
            "feature_sets": feature_sets,
            "threshold_fprs": threshold_fprs,
            "num_summary_rows": int(len(summary)),
            "num_per_generator_rows": int(len(per_gen)),
        },
        os.path.join(analysis_dir, "run_metadata.json"),
    )
    print(f"Wrote photo-only anomaly analysis under: {analysis_dir}")


if __name__ == "__main__":
    main()

