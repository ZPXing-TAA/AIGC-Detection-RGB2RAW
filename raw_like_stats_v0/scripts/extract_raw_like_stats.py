#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_utils import copy_config, load_config, resolve_path, update_run_metadata, write_json
from src.io_utils import (
    failed_images_path,
    failure_record,
    load_raw_like,
    manifest_path_from_config,
    raw_like_path,
    read_manifest,
    write_failed_images,
)
from src.raw_feature_extractor import extract_features


BASE_METADATA_COLUMNS = [
    "image_id",
    "dataset",
    "split",
    "subset",
    "generator",
    "generator_family",
    "label",
    "source",
    "input_path",
    "orig_ext",
    "raw_like_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract handcrafted RAW-like statistics.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature files.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug image limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    csv_path = resolve_path(cfg, cfg["features"]["output_csv"])
    parquet_path = resolve_path(cfg, cfg["features"]["output_parquet"])
    schema_path = resolve_path(cfg, cfg["features"]["schema_json"])

    if csv_path.exists() and parquet_path.exists() and schema_path.exists() and not args.overwrite:
        write_failed_images(failed_images_path(cfg), [], stage="feature_extraction")
        print(f"Feature files exist, skip: {csv_path} / {parquet_path}")
        return

    rows = read_manifest(manifest_path_from_config(cfg))
    if args.limit is not None:
        limited = []
        for source in ("photo", "gen"):
            limited.extend([row for row in rows if row["source"] == source][: args.limit])
        rows = limited

    feature_rows: List[dict] = []
    failed = []
    raw_shape = None
    raw_min = None
    raw_max = None

    for row in tqdm(rows, desc="features"):
        path = raw_like_path(cfg, row)
        if not path.exists():
            failed.append(failure_record("feature_extraction", row, "missing RAW-like file: {}".format(path)))
            continue
        try:
            raw = load_raw_like(path)
        except Exception as exc:
            failed.append(failure_record("feature_extraction", row, "failed to load RAW-like file: {}".format(exc)))
            continue
        raw_shape = list(raw.shape)
        cur_min = float(np.min(raw))
        cur_max = float(np.max(raw))
        raw_min = cur_min if raw_min is None else min(raw_min, cur_min)
        raw_max = cur_max if raw_max is None else max(raw_max, cur_max)
        payload = {
            "image_id": row["image_id"],
            "dataset": row.get("dataset", ""),
            "split": row.get("split", ""),
            "subset": row.get("subset", ""),
            "generator": row.get("generator", ""),
            "generator_family": row.get("generator_family", ""),
            "label": int(row["label"]),
            "source": row["source"],
            "input_path": row["path"],
            "orig_ext": row["orig_ext"],
            "raw_like_path": str(path),
        }
        try:
            payload.update(extract_features(raw, cfg["features"]))
        except Exception as exc:
            failed.append(failure_record("feature_extraction", row, "feature extraction failed: {}".format(exc)))
            continue
        feature_rows.append(payload)

    write_failed_images(failed_images_path(cfg), failed, stage="feature_extraction")
    if failed:
        raise RuntimeError(f"{len(failed)} images failed during feature extraction. See {failed_images_path(cfg)}")
    if not feature_rows:
        raise RuntimeError("No feature rows were extracted.")

    df = pd.DataFrame(feature_rows)
    feature_cols = [
        c
        for c in df.columns
        if c.startswith(("noise_", "hp_", "channel_", "freq_", "cfa_"))
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False)

    schema = {
        "raw_like_format": "packed_rggb",
        "raw_like_shape": raw_shape,
        "raw_like_value_range": [raw_min, raw_max],
        "metadata_columns": [c for c in BASE_METADATA_COLUMNS if c in df.columns],
        "feature_columns": feature_cols,
        "num_rows": len(df),
        "num_features": len(feature_cols),
        "feature_groups": {
            "noise": [c for c in feature_cols if c.startswith("noise_")],
            "highpass": [c for c in feature_cols if c.startswith("hp_")],
            "channel": [c for c in feature_cols if c.startswith("channel_")],
            "frequency": [c for c in feature_cols if c.startswith("freq_")],
            "cfa_like": [c for c in feature_cols if c.startswith("cfa_")],
        },
    }
    write_json(schema_path, schema)
    copy_config(cfg, resolve_path(cfg, "outputs"))
    update_run_metadata(
        cfg,
        raw_like_format="packed_rggb",
        raw_like_shape=raw_shape,
        raw_like_value_range=[raw_min, raw_max],
        num_feature_rows=len(df),
        num_features=len(feature_cols),
    )
    print(f"Wrote features: {csv_path}")
    print(f"Wrote parquet: {parquet_path}")
    print(f"Wrote schema: {schema_path}")


if __name__ == "__main__":
    main()
