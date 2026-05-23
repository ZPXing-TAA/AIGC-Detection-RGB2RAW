from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from .config_utils import ensure_dir


METADATA_COLUMNS = {
    "image_id",
    "dataset",
    "split",
    "subset",
    "generator",
    "generator_family",
    "label",
    "source",
    "raw_like_path",
    "orig_ext",
    "input_path",
}
FEATURE_PREFIXES = ("noise_", "hp_", "channel_", "freq_", "cfa_")


def feature_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for col in df.columns:
        if col in METADATA_COLUMNS:
            continue
        if col.startswith(FEATURE_PREFIXES) and pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def feature_group_columns(cols: Iterable[str]) -> Dict[str, List[str]]:
    groups = {
        "noise": [c for c in cols if c.startswith("noise_")],
        "highpass": [c for c in cols if c.startswith("hp_")],
        "channel": [c for c in cols if c.startswith("channel_")],
        "frequency": [c for c in cols if c.startswith("freq_")],
        "cfa_like": [c for c in cols if c.startswith("cfa_")],
        "all": list(cols),
    }
    return groups


def prepare_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    x = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    x = SimpleImputer(strategy="median").fit_transform(x)
    return x, y


def save_scatter(path: Path, coords: np.ndarray, labels: np.ndarray, title: str) -> None:
    ensure_dir(path.parent)
    plt.figure(figsize=(7, 6))
    for label, name, color in [(0, "photo", "#2f6f9f"), (1, "gen", "#c44e52")]:
        mask = labels == label
        plt.scatter(coords[mask, 0], coords[mask, 1], s=12, alpha=0.72, label=name, c=color)
    plt.title(title)
    plt.xlabel("dim 1")
    plt.ylabel("dim 2")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run_embedding_plots(x: np.ndarray, y: np.ndarray, out_dir: Path, seed: int) -> None:
    x_scaled = StandardScaler().fit_transform(x)
    if len(y) < 10:
        coords = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
        save_scatter(out_dir / "umap_raw_like.png", coords, y, "PCA RAW-like features")
        save_scatter(out_dir / "tsne_raw_like.png", coords, y, "PCA RAW-like features")
        return
    else:
        try:
            import umap

            coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(x_scaled)
        except Exception:
            coords = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
    save_scatter(out_dir / "umap_raw_like.png", coords, y, "UMAP RAW-like features")

    perplexity = max(2, min(30, (len(y) - 1) // 3))
    if len(y) > 3:
        tsne = TSNE(n_components=2, random_state=seed, perplexity=perplexity, init="pca", learning_rate=200.0)
        coords = tsne.fit_transform(x_scaled)
    else:
        coords = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
    save_scatter(out_dir / "tsne_raw_like.png", coords, y, "t-SNE RAW-like features")


def save_feature_histograms(df: pd.DataFrame, out_dir: Path, selected: Dict[str, List[str]]) -> None:
    root = out_dir / "feature_histograms"
    for group, cols in selected.items():
        group_dir = root / group
        ensure_dir(group_dir)
        for col in cols:
            if col not in df.columns:
                continue
            photo = df.loc[df["label"] == 0, col].replace([np.inf, -np.inf], np.nan).dropna()
            gen = df.loc[df["label"] == 1, col].replace([np.inf, -np.inf], np.nan).dropna()
            if photo.empty or gen.empty:
                continue
            plt.figure(figsize=(7, 4))
            plt.hist(photo, bins=40, alpha=0.6, density=True, label="photo", color="#2f6f9f")
            plt.hist(gen, bins=40, alpha=0.6, density=True, label="gen", color="#c44e52")
            plt.title(col)
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(group_dir / f"{col}.png", dpi=160)
            plt.close()


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> Dict[str, float]:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    try:
        out["auroc"] = float(roc_auc_score(y_true, scores))
    except Exception:
        out["auroc"] = 0.0
    try:
        out["average_precision"] = float(average_precision_score(y_true, scores))
    except Exception:
        out["average_precision"] = 0.0
    return out


def evaluate_classifier(model, x: np.ndarray, y: np.ndarray, seed: int, test_size: float) -> Dict[str, float]:
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=seed, stratify=y
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(x_test)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(x_test)
    else:
        scores = pred
    return metric_dict(y_test, pred, scores)


def run_linear_probe(x: np.ndarray, y: np.ndarray, seed: int, test_size: float) -> Dict[str, Dict[str, float]]:
    models = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        ),
        "linear_svm": make_pipeline(StandardScaler(), LinearSVC(class_weight="balanced", random_state=seed)),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            class_weight="balanced",
            n_jobs=4,
        ),
    }
    return {name: evaluate_classifier(model, x, y, seed, test_size) for name, model in models.items()}


def run_gmm_likelihood(x: np.ndarray, y: np.ndarray, out_dir: Path, seed: int, components: int) -> Dict[str, float]:
    photo = x[y == 0]
    gen = x[y == 1]
    if len(photo) < 2 or len(gen) < 1:
        return {
            "auroc_generated_as_anomaly": 0.0,
            "average_precision_generated_as_anomaly": 0.0,
            "photo_test_mean_loglik": 0.0,
            "gen_mean_loglik": 0.0,
            "gmm_components": 0,
            "skipped": True,
        }
    if len(photo) < 4:
        photo_train = photo
        photo_test = photo
    else:
        photo_train, photo_test = train_test_split(photo, test_size=0.3, random_state=seed)
    scaler = StandardScaler().fit(photo_train)
    photo_train = scaler.transform(photo_train)
    photo_test = scaler.transform(photo_test)
    gen_scaled = scaler.transform(gen)
    n_components = max(1, min(components, len(photo_train)))
    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=seed)
    gmm.fit(photo_train)
    score_photo = gmm.score_samples(photo_test)
    score_gen = gmm.score_samples(gen_scaled)
    y_true = np.concatenate([np.zeros_like(score_photo, dtype=np.int64), np.ones_like(score_gen, dtype=np.int64)])
    anomaly_score = -np.concatenate([score_photo, score_gen])

    plt.figure(figsize=(7, 4))
    plt.hist(score_photo, bins=40, alpha=0.65, density=True, label="photo", color="#2f6f9f")
    plt.hist(score_gen, bins=40, alpha=0.65, density=True, label="gen", color="#c44e52")
    plt.title("Photo-trained GMM log likelihood")
    plt.xlabel("log likelihood")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "gmm_likelihood_hist.png", dpi=160)
    plt.close()

    return {
        "auroc_generated_as_anomaly": float(roc_auc_score(y_true, anomaly_score)),
        "average_precision_generated_as_anomaly": float(average_precision_score(y_true, anomaly_score)),
        "photo_test_mean_loglik": float(np.mean(score_photo)),
        "gen_mean_loglik": float(np.mean(score_gen)),
        "gmm_components": int(n_components),
    }


def run_ablation(df: pd.DataFrame, cols: List[str], seed: int, test_size: float) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    groups = feature_group_columns(cols)
    y = df["label"].to_numpy(dtype=np.int64)
    for group, group_cols in groups.items():
        if not group_cols:
            continue
        x, _ = prepare_matrix(df, group_cols)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
        )
        out[group] = evaluate_classifier(model, x, y, seed, test_size)
        out[group]["num_features"] = len(group_cols)
    return out


def write_json(path: Path, payload: dict) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
