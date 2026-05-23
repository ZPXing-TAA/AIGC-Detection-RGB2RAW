#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_utils import copy_config, load_config, resolve_path, update_run_metadata, write_json
from src.io_utils import (
    build_aigibench_manifest,
    build_bfree_manifest,
    manifest_path_from_config,
    manifest_summary,
    manifest_summary_path,
    read_manifest,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare image manifest for RAW-like statistics.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument(
        "--dataset",
        choices=["bfree", "aigibench"],
        default=None,
        help="Dataset manifest type. Defaults to data.dataset_name from config.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite an existing manifest.")
    parser.add_argument("--limit-per-class", type=int, default=None, help="Optional debug cap per class.")
    parser.add_argument(
        "--max-per-label-per-subset",
        type=int,
        default=None,
        help="Optional AIGIBench smoke cap for each split/subset/label bucket.",
    )
    parser.add_argument("--max-train-per-label-per-subset", type=int, default=None, help="AIGIBench train cap per subset/label.")
    parser.add_argument("--max-val-per-label-per-subset", type=int, default=None, help="AIGIBench val cap per subset/label.")
    parser.add_argument("--max-test-per-label-per-subset", type=int, default=None, help="AIGIBench test cap per subset/label.")
    return parser.parse_args()


def _dataset_from_args(args: argparse.Namespace, cfg: dict) -> str:
    if args.dataset:
        return args.dataset
    name = cfg.get("data", {}).get("dataset_name", "bfree_online").lower()
    if "aigibench" in name:
        return "aigibench"
    return "bfree"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    dataset = _dataset_from_args(args, cfg)
    manifest_path = manifest_path_from_config(cfg)
    if manifest_path.exists() and not args.overwrite:
        rows = read_manifest(manifest_path)
        write_json(manifest_summary_path(cfg), manifest_summary(rows))
        print(f"Manifest exists, skip: {manifest_path} ({len(rows)} rows). Use --overwrite to rebuild.")
        print(f"Wrote manifest summary: {manifest_summary_path(cfg)}")
        return

    data_cfg = cfg["data"]
    max_per_class = args.limit_per_class
    if max_per_class is None:
        max_per_class = data_cfg.get("max_per_class")

    if dataset == "aigibench":
        max_per_label_per_subset = args.max_per_label_per_subset
        if max_per_label_per_subset is None:
            max_per_label_per_subset = data_cfg.get("max_per_label_per_subset")
        split_limits = dict(data_cfg.get("max_per_label_per_subset_by_split") or {})
        if args.max_train_per_label_per_subset is not None:
            split_limits["train"] = args.max_train_per_label_per_subset
        if args.max_val_per_label_per_subset is not None:
            split_limits["val"] = args.max_val_per_label_per_subset
        if args.max_test_per_label_per_subset is not None:
            split_limits["test"] = args.max_test_per_label_per_subset
        rows = build_aigibench_manifest(
            root=data_cfg["aigibench_root"],
            extensions=data_cfg["image_extensions"],
            seed=int(cfg["project"].get("random_seed", 42)),
            train_subsets=data_cfg.get("aigibench_train_subsets"),
            val_subsets=data_cfg.get("aigibench_val_subsets"),
            test_subsets=data_cfg.get("aigibench_test_subsets"),
            generator_families=data_cfg.get("generator_families", {}),
            max_per_label_per_subset=max_per_label_per_subset,
            max_per_label_per_subset_by_split=split_limits,
        )
    else:
        rows = build_bfree_manifest(
            real_dir=data_cfg["bfree_real_dir"],
            fake_dir=data_cfg["bfree_fake_dir"],
            extensions=data_cfg["image_extensions"],
            seed=int(cfg["project"].get("random_seed", 42)),
            balance_classes=bool(data_cfg.get("balance_classes", True)),
            max_per_class=max_per_class,
        )
    write_manifest(manifest_path, rows)
    summary = manifest_summary(rows)
    write_json(manifest_summary_path(cfg), summary)
    counts = Counter(row["source"] for row in rows)
    print(f"Wrote manifest: {manifest_path}")
    print(f"Counts: {dict(counts)}")
    print(f"Wrote manifest summary: {manifest_summary_path(cfg)}")

    copy_config(cfg, resolve_path(cfg, "outputs"))
    update_run_metadata(
        cfg,
        dataset_name=data_cfg.get("dataset_name", "bfree_online"),
        manifest_path=str(manifest_path),
        manifest_summary_path=str(manifest_summary_path(cfg)),
        num_photo_images=int(counts.get("photo", 0)),
        num_gen_images=int(counts.get("gen", 0)),
        image_size=int(cfg["preprocess"]["image_size"]),
    )


if __name__ == "__main__":
    main()
