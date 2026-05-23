#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.raw_cache import save_raw_like
from src.utils.config import ensure_dir, load_config
from src.utils.cycleisp_wrapper import CycleISPRgb2Raw
from src.utils.image_ops import load_rgb_chw, resize_chw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate RAW-like cache with CycleISP.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "all"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of disjoint manifest shards.")
    parser.add_argument("--shard-index", type=int, default=0, help="Zero-based shard index to process.")
    return parser.parse_args()


def batched(items: List[Dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def write_failed(path: Path, rows: List[Dict[str, object]]) -> None:
    ensure_dir(path.parent)
    fieldnames = ["stage", "image_id", "split", "subset", "generator", "label", "path", "raw_like_path", "error"]
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    manifest_path = Path(cfg["data"]["manifest_path"])
    if not manifest_path.exists():
        raise FileNotFoundError("Manifest not found. Run generate_full_manifest.py first: {}".format(manifest_path))
    df = pd.read_csv(str(manifest_path))
    if args.split != "all":
        df = df[df["split"] == args.split].copy()
    df = df.sort_values("image_id").reset_index(drop=True)
    if int(args.num_shards) < 1:
        raise ValueError("--num-shards must be >= 1")
    if int(args.shard_index) < 0 or int(args.shard_index) >= int(args.num_shards):
        raise ValueError("--shard-index must be in [0, num_shards)")
    if int(args.num_shards) > 1:
        shard_mask = (np.arange(len(df)) % int(args.num_shards)) == int(args.shard_index)
        df = df[shard_mask].copy().reset_index(drop=True)
    if args.limit is not None:
        df = df.iloc[: int(args.limit)].copy()

    rows = df.to_dict("records")
    pending = [r for r in rows if args.overwrite or not Path(str(r["raw_like_path"])).exists()]
    shard_name = args.split
    if int(args.num_shards) > 1:
        shard_name = "{}_shard{}of{}".format(args.split, int(args.shard_index), int(args.num_shards))
    print("Split={} total={} pending={}".format(shard_name, len(rows), len(pending)))
    if not pending:
        return

    batch_size = int(args.batch_size or cfg["cycleisp"].get("batch_size", 1))
    device = args.device or cfg["cycleisp"].get("device", "cuda:0")
    input_size = int(cfg["cycleisp"].get("input_size", int(cfg["data"]["raw_cache_size"]) * 2))
    raw_size = int(cfg["data"]["raw_cache_size"])
    raw_dtype = str(cfg["data"].get("raw_cache_dtype", "uint16"))
    raw_layout = str(cfg["data"].get("raw_cache_layout", "chw"))

    model = CycleISPRgb2Raw(
        repo_dir=cfg["cycleisp"]["repo_dir"],
        checkpoint_path=cfg["cycleisp"]["checkpoint_path"],
        device=device,
    )

    failed: List[Dict[str, object]] = []
    raw_min = None
    raw_max = None
    raw_shape = None
    wrote = 0
    for batch_rows in tqdm(
        batched(pending, batch_size),
        total=(len(pending) + batch_size - 1) // batch_size,
        desc=shard_name,
    ):
        arrays = []
        valid_rows = []
        for row in batch_rows:
            try:
                arrays.append(load_rgb_chw(str(row["path"]), input_size))
                valid_rows.append(row)
            except Exception as exc:
                row = dict(row)
                row["stage"] = "load_rgb"
                row["error"] = str(exc)
                failed.append(row)
        if not arrays:
            continue
        try:
            outputs = model(arrays)
        except Exception as exc:
            for row in valid_rows:
                row = dict(row)
                row["stage"] = "cycleisp"
                row["error"] = str(exc)
                failed.append(row)
            continue
        for row, raw in zip(valid_rows, outputs):
            try:
                raw = np.asarray(raw, dtype=np.float32)
                raw = resize_chw(raw, raw_size)
                save_raw_like(str(row["raw_like_path"]), raw, dtype=raw_dtype, layout=raw_layout)
                raw_shape = list(raw.shape)
                cur_min = float(np.min(raw))
                cur_max = float(np.max(raw))
                raw_min = cur_min if raw_min is None else min(raw_min, cur_min)
                raw_max = cur_max if raw_max is None else max(raw_max, cur_max)
                wrote += 1
            except Exception as exc:
                row = dict(row)
                row["stage"] = "save_raw_like"
                row["error"] = str(exc)
                failed.append(row)

    out_dir = Path(cfg["project"]["output_dir"]) / "raw_cache_logs"
    ensure_dir(out_dir)
    failed_path = out_dir / "failed_images_{}.csv".format(shard_name)
    write_failed(failed_path, failed)
    metadata = {
        "created_at": datetime.now().isoformat(),
        "split": args.split,
        "shard_name": shard_name,
        "num_shards": int(args.num_shards),
        "shard_index": int(args.shard_index),
        "total_rows": int(len(rows)),
        "pending": int(len(pending)),
        "wrote": int(wrote),
        "failed": int(len(failed)),
        "raw_cache_dtype": raw_dtype,
        "raw_cache_layout": raw_layout,
        "raw_cache_size": raw_size,
        "cycleisp_input_size": input_size,
        "raw_like_shape": raw_shape,
        "raw_like_value_range": [raw_min, raw_max],
    }
    with open(str(out_dir / "metadata_{}.json".format(shard_name)), "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    if failed:
        raise RuntimeError("{} images failed; see {}".format(len(failed), failed_path))
    print("Finished split={}, wrote {} RAW-like arrays.".format(shard_name, wrote))


if __name__ == "__main__":
    main()
