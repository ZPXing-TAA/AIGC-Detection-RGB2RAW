#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis_utils import (
    feature_columns,
    prepare_matrix,
    run_ablation,
    run_embedding_plots,
    run_gmm_likelihood,
    run_linear_probe,
    save_feature_histograms,
    write_json,
)
from src.config_utils import copy_config, ensure_dir, load_config, resolve_path, update_run_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RAW-like feature distributions.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--features", required=True, help="Feature parquet or CSV path.")
    parser.add_argument("--overwrite", action="store_true", help="Accepted for run_all symmetry.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    features_path = resolve_path(cfg, args.features)
    out_dir = resolve_path(cfg, cfg["analysis"]["output_dir"])
    ensure_dir(out_dir)
    expected_outputs = [
        out_dir / "umap_raw_like.png",
        out_dir / "tsne_raw_like.png",
        out_dir / "gmm_likelihood_hist.png",
        out_dir / "gmm_likelihood_results.json",
        out_dir / "linear_probe_results.json",
        out_dir / "feature_ablation_results.json",
    ]
    if all(path.exists() for path in expected_outputs) and not args.overwrite:
        print(f"Analysis outputs exist, skip: {out_dir}. Use --overwrite to rebuild.")
        return

    if features_path.suffix.lower() == ".csv":
        df = pd.read_csv(features_path)
    else:
        df = pd.read_parquet(features_path)

    cols = feature_columns(df)
    if not cols:
        raise RuntimeError("No numeric RAW-like feature columns found.")
    x, y = prepare_matrix(df, cols)
    seed = int(cfg["analysis"].get("random_seed", cfg["project"].get("random_seed", 42)))
    test_size = float(cfg["analysis"].get("test_size", 0.3))

    run_embedding_plots(x, y, out_dir, seed)
    save_feature_histograms(df, out_dir, cfg["analysis"].get("representative_features", {}))

    gmm_results = run_gmm_likelihood(
        x,
        y,
        out_dir,
        seed=seed,
        components=int(cfg["analysis"].get("gmm_components", 5)),
    )
    write_json(out_dir / "gmm_likelihood_results.json", gmm_results)

    linear_results = run_linear_probe(x, y, seed=seed, test_size=test_size)
    write_json(out_dir / "linear_probe_results.json", linear_results)

    ablation_results = run_ablation(df, cols, seed=seed, test_size=test_size)
    write_json(out_dir / "feature_ablation_results.json", ablation_results)

    copy_config(cfg, resolve_path(cfg, "outputs"))
    update_run_metadata(cfg, analysis_output_dir=str(out_dir), num_analysis_features=len(cols))
    print(f"Wrote analysis outputs under: {out_dir}")


if __name__ == "__main__":
    main()
