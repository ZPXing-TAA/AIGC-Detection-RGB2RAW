#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_utils import copy_config, load_config, resolve_path, update_run_metadata
from src.cycleisp_wrapper import CycleISPRgb2Raw
from src.image_preprocess import load_preprocessed_rgb
from src.io_utils import (
    ensure_dir,
    failed_images_path,
    failure_record,
    manifest_path_from_config,
    preview_path,
    raw_like_path,
    read_manifest,
    save_bayer_preview,
    save_raw_like,
    split_rows,
    write_failed_images,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CycleISP RGB2RAW on manifest images.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--split",
        required=True,
        choices=["photo", "gen", "all", "train", "val", "test"],
        help="Manifest source split or official dataset split to process.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing RAW-like outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional debug image limit.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override cycleisp.batch_size.")
    parser.add_argument("--device", default=None, help="Override cycleisp.device, e.g. cuda:9.")
    return parser.parse_args()


def batched(items: List[dict], batch_size: int):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    rows = split_rows(read_manifest(manifest_path_from_config(cfg)), args.split)
    if args.limit is not None:
        rows = rows[: args.limit]

    pending = [row for row in rows if args.overwrite or not raw_like_path(cfg, row).exists()]
    print(f"Split={args.split} total={len(rows)} pending={len(pending)}")
    if not pending:
        write_failed_images(failed_images_path(cfg), [], stage="cycleisp")
        return

    for row in pending:
        ensure_dir(raw_like_path(cfg, row).parent)
        ensure_dir(preview_path(cfg, row).parent)

    model = CycleISPRgb2Raw(
        repo_dir=cfg["cycleisp"]["repo_dir"],
        checkpoint_path=cfg["cycleisp"]["checkpoint_path"],
        device=args.device or cfg["cycleisp"].get("device", "cuda:0"),
    )

    image_size = int(cfg["preprocess"]["image_size"])
    force_rgb = bool(cfg["preprocess"].get("force_rgb", True))
    raw_shape = None
    raw_min = None
    raw_max = None
    failed = []
    batch_size = int(args.batch_size if args.batch_size is not None else cfg["cycleisp"].get("batch_size", 1))

    for batch_rows in tqdm(batched(pending, batch_size), total=(len(pending) + batch_size - 1) // batch_size, desc=args.split):
        batch_arrays = []
        valid_rows = []
        for row in batch_rows:
            try:
                batch_arrays.append(load_preprocessed_rgb(row["path"], image_size=image_size, force_rgb=force_rgb))
                valid_rows.append(row)
            except Exception as exc:
                failed.append(failure_record("cycleisp", row, str(exc)))
        if not batch_arrays:
            continue

        try:
            outputs = model(batch_arrays)
        except Exception as exc:
            for row in valid_rows:
                failed.append(failure_record("cycleisp", row, "CycleISP batch failed: {}".format(exc)))
            continue
        if len(outputs) != len(valid_rows):
            for row in valid_rows:
                failed.append(
                    failure_record(
                        "cycleisp",
                        row,
                        "CycleISP output count mismatch: expected {}, got {}".format(len(valid_rows), len(outputs)),
                    )
                )
            continue
        for row, raw in zip(valid_rows, outputs):
            raw = np.asarray(raw, dtype=np.float32)
            save_raw_like(raw_like_path(cfg, row), raw)
            if bool(cfg["raw_like_output"].get("save_preview_png", True)):
                save_bayer_preview(preview_path(cfg, row), raw)
            raw_shape = list(raw.shape)
            cur_min = float(np.min(raw))
            cur_max = float(np.max(raw))
            raw_min = cur_min if raw_min is None else min(raw_min, cur_min)
            raw_max = cur_max if raw_max is None else max(raw_max, cur_max)

    if failed:
        fail_path = failed_images_path(cfg)
        write_failed_images(fail_path, failed, stage="cycleisp")
        raise RuntimeError(f"{len(failed)} images failed. See {fail_path}")
    write_failed_images(failed_images_path(cfg), [], stage="cycleisp")

    copy_config(cfg, resolve_path(cfg, "outputs"))
    update_run_metadata(
        cfg,
        raw_like_format="packed_rggb",
        raw_like_shape=raw_shape,
        raw_like_value_range=[raw_min, raw_max],
        last_processed_split=args.split,
    )
    print(f"Finished {args.split}: wrote {len(pending)} RAW-like arrays.")


if __name__ == "__main__":
    main()
