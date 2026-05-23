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

from src.config_utils import copy_config, ensure_dir, load_config, set_seed, utc_now_iso, write_json
from src.features.photo_only_compact import extract_photo_only_compact_features
from src.io_utils import load_raw_like, metadata_from_row, raw_like_path, read_manifest, sample_manifest


def feature_columns(df: pd.DataFrame) -> List[str]:
    prefixes = ("noise20_", "ai64_", "ai128_", "ai300_", "srm_noq_", "srm_q255_")
    return [c for c in df.columns if c.startswith(prefixes)]


def build_manifest_summary(df: pd.DataFrame) -> Dict[str, Any]:
    group_cols = [c for c in ["split", "subset", "label"] if c in df.columns]
    counts = []
    for keys, part in df.groupby(group_cols, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["count"] = int(len(part))
        counts.append(row)
    return {
        "total": int(len(df)),
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().sort_index().items()},
        "source_counts": {str(k): int(v) for k, v in df["source"].value_counts().items()} if "source" in df.columns else {},
        "counts_by_split_subset_label": counts,
    }


def feature_schema(df: pd.DataFrame) -> Dict[str, Any]:
    cols = feature_columns(df)
    groups = {
        "noise20": [c for c in cols if c.startswith("noise20_")],
        "ai64": [c for c in cols if c.startswith("ai64_")],
        "ai128": [c for c in cols if c.startswith("ai128_")],
        "ai300": [c for c in cols if c.startswith("ai300_")],
        "srm_noq": [c for c in cols if c.startswith("srm_noq_")],
        "srm_q255": [c for c in cols if c.startswith("srm_q255_")],
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
    parquet_path = os.path.join(out_dir, "features.parquet")
    csv_path = os.path.join(features_dir, "photo_only_compact_features.csv")
    failed_path = os.path.join(out_dir, "failed_images.csv")
    if os.path.exists(parquet_path) and not args.overwrite:
        print(f"Features already exist, skip: {parquet_path}")
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
    feat_cfg = config.get("feature_extraction", {})
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="photo_only_features"):
        meta = metadata_from_row(row)
        npy_path = raw_like_path(row, config)
        try:
            raw_like = load_raw_like(npy_path)
            feats = extract_photo_only_compact_features(raw_like, feat_cfg)
            rows.append({**meta, "raw_like_path": npy_path, **feats})
        except Exception as exc:
            failures.append({**meta, "raw_like_path": npy_path, "error": repr(exc)})

    failed = pd.DataFrame(failures)
    if failed.empty:
        failed = pd.DataFrame(columns=["image_id", "split", "subset", "label", "source", "path", "raw_like_path", "error"])
    failed.to_csv(failed_path, index=False)
    if not rows:
        raise RuntimeError(f"all feature extraction failed; failures written to {failed_path}")

    df = pd.DataFrame(rows)
    cols = feature_columns(df)
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    write_json(feature_schema(df), os.path.join(out_dir, "feature_schema.json"))
    write_json(
        {
            "created_at": utc_now_iso(),
            "protocol": "photo_only_anomaly_detection_v2",
            "manifest": manifest_path,
            "num_manifest_rows": int(len(manifest)),
            "num_feature_rows": int(len(df)),
            "num_failed": int(len(failures)),
            "num_features": int(len(cols)),
            "feature_rule": "compact variants only; not v1 full 817-dimensional default",
        },
        os.path.join(out_dir, "run_metadata.json"),
    )
    print(f"Wrote features: {parquet_path}")
    print(f"Wrote failed images: {failed_path}")


if __name__ == "__main__":
    main()

