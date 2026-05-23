#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import copy_config, ensure_dir, load_config, output_path, set_seed, utc_now_iso, write_json
from src.features.extractor import extract_all_features
from src.io_utils import load_raw_like, metadata_from_row, raw_like_path, read_manifest, sample_manifest, stable_feature_columns


def build_manifest_summary(df: pd.DataFrame) -> Dict[str, Any]:
    group_cols = [c for c in ["split", "subset", "label"] if c in df.columns]
    counts = []
    if group_cols:
        for keys, part in df.groupby(group_cols, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {col: value for col, value in zip(group_cols, keys)}
            row["count"] = int(len(part))
            counts.append(row)
    source_counts = df["source"].value_counts().to_dict() if "source" in df.columns else {}
    return {
        "total": int(len(df)),
        "source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "counts_by_split_subset_label": counts,
    }


def feature_schema(df: pd.DataFrame) -> Dict[str, Any]:
    cols = stable_feature_columns(df)
    groups = {
        "noise": [c for c in cols if c.startswith("noise_")],
        "srm": [c for c in cols if c.startswith("srm_")],
        "glcm": [c for c in cols if c.startswith("glcm_")],
        "frequency": [c for c in cols if c.startswith("freq_")],
    }
    return {
        "num_features": int(len(cols)),
        "feature_columns": cols,
        "groups": {k: {"count": len(v), "columns": v} for k, v in groups.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-per-split-label", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir:
        config["paths"]["output_dir"] = args.output_dir
    set_seed(int(config.get("project", {}).get("seed", 42)))
    out_dir = ensure_dir(config["paths"]["output_dir"])
    features_dir = ensure_dir(os.path.join(out_dir, "features"))
    features_csv = os.path.join(features_dir, "features.csv")
    features_parquet = os.path.join(out_dir, "features.parquet")
    failed_path = os.path.join(out_dir, config.get("feature_extraction", {}).get("failed_images_path", "failed_images.csv"))

    if os.path.exists(features_parquet) and not args.overwrite:
        print(f"Features already exist, skip: {features_parquet}")
        return

    copy_config(args.config, out_dir)
    manifest_path = args.manifest or config["paths"]["v0_manifest"]
    manifest = read_manifest(manifest_path)
    manifest = sample_manifest(
        manifest,
        limit_per_split_label=args.limit_per_split_label,
        seed=int(config.get("project", {}).get("seed", 42)),
    )
    write_json(build_manifest_summary(manifest), os.path.join(out_dir, "manifest_summary.json"))

    rows: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    feature_config = config.get("feature_extraction", {})
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="statistical_probes"):
        meta = metadata_from_row(row)
        npy_path = raw_like_path(row, config)
        try:
            raw_like = load_raw_like(npy_path)
            feats = extract_all_features(raw_like, feature_config)
            rows.append({**meta, "raw_like_path": npy_path, **feats})
        except Exception as exc:
            failures.append({**meta, "raw_like_path": npy_path, "error": repr(exc)})

    failed_df = pd.DataFrame(failures)
    if failed_df.empty:
        failed_df = pd.DataFrame(columns=["image_id", "split", "subset", "label", "source", "path", "raw_like_path", "error"])
    failed_df.to_csv(failed_path, index=False)

    if not rows:
        raise RuntimeError(f"all feature extraction failed; failures written to {failed_path}")
    df = pd.DataFrame(rows)
    feature_cols = stable_feature_columns(df)
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.to_csv(features_csv, index=False)
    df.to_parquet(features_parquet, index=False)
    write_json(feature_schema(df), os.path.join(out_dir, "feature_schema.json"))
    write_json(
        {
            "created_at": utc_now_iso(),
            "config": os.path.abspath(args.config),
            "manifest": manifest_path,
            "output_dir": out_dir,
            "num_manifest_rows": int(len(manifest)),
            "num_feature_rows": int(len(df)),
            "num_failed": int(len(failures)),
            "num_features": int(len(feature_cols)),
            "raw_like_cache": config["paths"]["raw_like_cache"],
        },
        os.path.join(out_dir, "run_metadata.json"),
    )
    print(f"Wrote features: {features_parquet}")
    print(f"Wrote failed images: {failed_path}")


if __name__ == "__main__":
    main()

