#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis_utils import run_gmm_likelihood, save_scatter, write_json
from src.config_utils import copy_config, ensure_dir, load_config, resolve_path, update_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze learned RAW-like embeddings with UMAP/t-SNE/GMM.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--embeddings",
        default=None,
        help="Learned embeddings parquet/csv. Defaults to learned.embeddings_parquet from config.",
    )
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to learned_analysis.output_dir.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing learned analysis outputs.")
    return parser.parse_args()


def learned_embedding_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("learned_emb_") and pd.api.types.is_numeric_dtype(df[c])]


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def make_matrix(df: pd.DataFrame, cols: List[str]):
    x = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float64)
    y = df["label"].to_numpy(dtype=np.int64)
    x = SimpleImputer(strategy="median").fit_transform(x)
    return x, y


def plot_umap(x_scaled: np.ndarray, y: np.ndarray, out_path: Path, seed: int) -> str:
    try:
        import umap

        coords = umap.UMAP(n_components=2, random_state=seed).fit_transform(x_scaled)
        method = "UMAP"
    except Exception:
        coords = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
        method = "PCA fallback"
    save_scatter(out_path, coords, y, "{} learned embeddings".format(method))
    return method


def plot_tsne(x_scaled: np.ndarray, y: np.ndarray, out_path: Path, seed: int) -> str:
    if len(y) <= 3:
        coords = PCA(n_components=2, random_state=seed).fit_transform(x_scaled)
        method = "PCA fallback"
    else:
        perplexity = max(2, min(30, (len(y) - 1) // 3))
        coords = TSNE(
            n_components=2,
            random_state=seed,
            perplexity=perplexity,
            init="pca",
            learning_rate=200.0,
        ).fit_transform(x_scaled)
        method = "t-SNE"
    save_scatter(out_path, coords, y, "{} learned embeddings".format(method))
    return method


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    embeddings_path = resolve_path(cfg, args.embeddings or cfg["learned"]["embeddings_parquet"])
    out_dir = resolve_path(
        cfg,
        args.output_dir or cfg.get("learned_analysis", {}).get("output_dir", "outputs/learned_analysis"),
    )
    ensure_dir(out_dir)

    expected = [
        out_dir / "umap_learned.png",
        out_dir / "tsne_learned.png",
        out_dir / "gmm_likelihood_hist.png",
        out_dir / "gmm_likelihood_results.json",
        out_dir / "learned_analysis_summary.json",
    ]
    if all(path.exists() for path in expected) and not args.overwrite:
        print("Learned analysis outputs exist, skip: {}. Use --overwrite to rebuild.".format(out_dir))
        return

    df = load_table(embeddings_path)
    cols = learned_embedding_columns(df)
    if not cols:
        raise RuntimeError("No learned_emb_* numeric columns found in {}".format(embeddings_path))
    x, y = make_matrix(df, cols)
    seed = int(cfg.get("learned_analysis", {}).get("random_seed", cfg["learned"].get("random_seed", 42)))
    components = int(cfg.get("learned_analysis", {}).get("gmm_components", cfg["analysis"].get("gmm_components", 5)))

    x_scaled = StandardScaler().fit_transform(x)
    umap_method = plot_umap(x_scaled, y, out_dir / "umap_learned.png", seed)
    tsne_method = plot_tsne(x_scaled, y, out_dir / "tsne_learned.png", seed)
    gmm_results = run_gmm_likelihood(x, y, out_dir, seed=seed, components=components)
    write_json(out_dir / "gmm_likelihood_results.json", gmm_results)

    summary = {
        "embeddings_path": str(embeddings_path),
        "num_rows": int(len(df)),
        "embedding_dim": int(len(cols)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "umap_method": umap_method,
        "tsne_method": tsne_method,
        "gmm_results": gmm_results,
        "output_files": [str(path) for path in expected],
    }
    write_json(out_dir / "learned_analysis_summary.json", summary)
    copy_config(cfg, resolve_path(cfg, "outputs"))
    update_run_metadata(cfg, learned_analysis_output_dir=str(out_dir), learned_analysis_embedding_dim=len(cols))
    print("Wrote learned analysis outputs under: {}".format(out_dir))


if __name__ == "__main__":
    main()
