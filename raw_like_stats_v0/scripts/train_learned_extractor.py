#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_key, "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, average_precision_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config_utils import copy_config, ensure_dir, load_config, resolve_path, update_run_metadata, write_json
from src.io_utils import manifest_path_from_config, raw_like_path, read_manifest
from src.learned_data import (
    RawLikeDataset,
    limit_per_class,
    rows_to_split_records,
    split_rows_stratified,
    validate_raw_outputs,
)
from src.learned_extractor import RawLikeCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a supervised learned RAW-like feature extractor.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing learned outputs.")
    parser.add_argument("--limit", type=int, default=None, help="Optional per-class debug limit.")
    parser.add_argument("--epochs", type=int, default=None, help="Override learned.train.epochs.")
    parser.add_argument("--device", default=None, help="Override learned.device.")
    parser.add_argument("--extract-only", action="store_true", help="Only export embeddings from an existing checkpoint.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learned_paths(cfg: dict) -> Dict[str, Path]:
    out_dir = resolve_path(cfg, cfg["learned"]["output_dir"])
    return {
        "out_dir": out_dir,
        "checkpoint_dir": out_dir / "checkpoints",
        "best_checkpoint": resolve_path(cfg, cfg["learned"]["checkpoint_path"]),
        "last_checkpoint": out_dir / "checkpoints" / "last.pt",
        "metrics": out_dir / "learned_metrics.json",
        "curves": out_dir / "training_curves.png",
        "splits": out_dir / "learned_splits.csv",
        "embeddings_csv": resolve_path(cfg, cfg["learned"]["embeddings_csv"]),
        "embeddings_parquet": resolve_path(cfg, cfg["learned"]["embeddings_parquet"]),
        "schema": resolve_path(cfg, cfg["learned"]["schema_json"]),
    }


def make_model(cfg: dict) -> RawLikeCNN:
    model_cfg = cfg["learned"]["model"]
    return RawLikeCNN(
        in_channels=int(model_cfg.get("in_channels", 4)),
        base_channels=int(model_cfg.get("base_channels", 32)),
        embedding_dim=int(model_cfg.get("embedding_dim", 64)),
        dropout=float(model_cfg.get("dropout", 0.1)),
    )


def make_loader(cfg: dict, rows: Sequence[dict], split: str, train_cfg: dict) -> DataLoader:
    dataset = RawLikeDataset(
        cfg,
        rows,
        augment_hflip=bool(train_cfg.get("augment_hflip", False)) and split == "train",
    )
    return DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch_size", 16)),
        shuffle=(split == "train"),
        num_workers=int(train_cfg.get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )


def binary_metrics(labels: np.ndarray, scores: np.ndarray, loss: float) -> Dict[str, float]:
    if labels.size == 0:
        return {"loss": float(loss), "accuracy": 0.0, "balanced_accuracy": 0.0, "auroc": 0.0, "average_precision": 0.0}
    preds = (scores >= 0.5).astype(np.int64)
    out = {
        "loss": float(loss),
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
    }
    if len(np.unique(labels)) > 1:
        out["auroc"] = float(roc_auc_score(labels, scores))
        out["average_precision"] = float(average_precision_score(labels, scores))
    else:
        out["auroc"] = 0.0
        out["average_precision"] = 0.0
    return out


def run_epoch(model, loader, criterion, device, optimizer=None) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0
    labels: List[np.ndarray] = []
    scores: List[np.ndarray] = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        if training:
            loss.backward()
            optimizer.step()

        batch_size = int(y.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
        labels.append(y.detach().cpu().numpy().astype(np.int64))
        scores.append(torch.sigmoid(logits).detach().cpu().numpy())

    if total_count == 0:
        return binary_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float32), 0.0)
    y_true = np.concatenate(labels)
    y_score = np.concatenate(scores)
    return binary_metrics(y_true, y_score, total_loss / float(total_count))


def save_checkpoint(path: Path, model, optimizer, epoch: int, metrics: dict, cfg: dict) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "metrics": metrics,
            "learned_config": cfg["learned"],
        },
        str(path),
    )


def load_checkpoint(path: Path, model, device) -> dict:
    checkpoint = torch.load(str(path), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def plot_curves(path: Path, history: List[dict]) -> None:
    ensure_dir(path.parent)
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [h["train"]["loss"] for h in history], label="train loss")
    plt.plot(epochs, [h["val"]["loss"] for h in history], label="val loss")
    plt.plot(epochs, [h["val"]["auroc"] for h in history], label="val auroc")
    plt.xlabel("epoch")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def export_embeddings(cfg: dict, model, rows: Sequence[dict], device, out_csv: Path, out_parquet: Path, schema_path: Path) -> None:
    loader = DataLoader(
        RawLikeDataset(cfg, rows, augment_hflip=False),
        batch_size=int(cfg["learned"]["train"].get("batch_size", 16)),
        shuffle=False,
        num_workers=int(cfg["learned"]["train"].get("num_workers", 2)),
        pin_memory=torch.cuda.is_available(),
    )
    model.eval()
    records: List[dict] = []
    with torch.no_grad():
        for x, _, image_ids in loader:
            x = x.to(device, non_blocking=True)
            logits, emb = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            emb_np = emb.detach().cpu().numpy()
            row_by_id = {row["image_id"]: row for row in rows}
            for idx, image_id in enumerate(image_ids):
                row = row_by_id[str(image_id)]
                record = {
                    "image_id": row["image_id"],
                    "label": int(row["label"]),
                    "source": row["source"],
                    "input_path": row["path"],
                    "orig_ext": row["orig_ext"],
                    "raw_like_path": str(raw_like_path(cfg, row)),
                    "learned_score": float(probs[idx]),
                    "learned_pred": int(probs[idx] >= 0.5),
                }
                for j in range(emb_np.shape[1]):
                    record["learned_emb_{:03d}".format(j)] = float(emb_np[idx, j])
                records.append(record)

    df = pd.DataFrame(records)
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)
    df.to_parquet(out_parquet, index=False)
    emb_cols = [c for c in df.columns if c.startswith("learned_emb_")]
    write_json(
        schema_path,
        {
            "num_rows": len(df),
            "embedding_dim": len(emb_cols),
            "metadata_columns": ["image_id", "label", "source", "input_path", "orig_ext", "raw_like_path"],
            "score_columns": ["learned_score", "learned_pred"],
            "embedding_columns": emb_cols,
        },
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if "learned" not in cfg:
        raise KeyError("Missing learned section in config.")

    paths = learned_paths(cfg)
    expected = [paths["best_checkpoint"], paths["metrics"], paths["embeddings_parquet"], paths["schema"]]
    if all(path.exists() for path in expected) and not args.overwrite and not args.extract_only:
        print("Learned outputs exist, skip: {}. Use --overwrite to rebuild.".format(paths["out_dir"]))
        return

    learned_cfg = cfg["learned"]
    train_cfg = learned_cfg["train"]
    seed = int(learned_cfg.get("random_seed", cfg["project"].get("random_seed", 42)))
    set_seed(seed)
    device_name = args.device or learned_cfg.get("device", cfg["cycleisp"].get("device", "cuda:0"))
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    rows = read_manifest(manifest_path_from_config(cfg))
    if args.limit is not None:
        rows = limit_per_class(rows, args.limit)
    validate_raw_outputs(cfg, rows)

    splits = split_rows_stratified(
        rows,
        seed=seed,
        train_ratio=float(train_cfg.get("train_ratio", 0.7)),
        val_ratio=float(train_cfg.get("val_ratio", 0.15)),
        test_ratio=float(train_cfg.get("test_ratio", 0.15)),
    )
    ensure_dir(paths["out_dir"])
    pd.DataFrame(rows_to_split_records(splits)).to_csv(paths["splits"], index=False)

    model = make_model(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("lr", 0.001)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0001)),
    )

    if args.extract_only:
        load_checkpoint(paths["best_checkpoint"], model, device)
        export_embeddings(cfg, model, rows, device, paths["embeddings_csv"], paths["embeddings_parquet"], paths["schema"])
        print("Exported learned embeddings from existing checkpoint.")
        return

    loaders = {
        "train": make_loader(cfg, splits["train"], "train", train_cfg),
        "val": make_loader(cfg, splits["val"], "val", train_cfg),
        "test": make_loader(cfg, splits["test"], "test", train_cfg),
    }
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 20))
    patience = int(train_cfg.get("patience", 8))
    best_metric = -1.0
    best_epoch = 0
    stale = 0
    history: List[dict] = []

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(model, loaders["train"], criterion, device, optimizer)
        with torch.no_grad():
            val_metrics = run_epoch(model, loaders["val"], criterion, device, optimizer=None)
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        print(
            "epoch={:03d} train_loss={:.4f} val_loss={:.4f} val_auroc={:.4f}".format(
                epoch, train_metrics["loss"], val_metrics["loss"], val_metrics["auroc"]
            )
        )
        metric = val_metrics.get("auroc", 0.0)
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch
            stale = 0
            save_checkpoint(paths["best_checkpoint"], model, optimizer, epoch, row, cfg)
        else:
            stale += 1
            if stale >= patience:
                print("Early stopping at epoch {}.".format(epoch))
                break
        save_checkpoint(paths["last_checkpoint"], model, optimizer, epoch, row, cfg)

    load_checkpoint(paths["best_checkpoint"], model, device)
    with torch.no_grad():
        test_metrics = run_epoch(model, loaders["test"], criterion, device, optimizer=None)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_auroc": best_metric,
        "test": test_metrics,
        "history": history,
        "split_counts": {name: len(split_rows) for name, split_rows in splits.items()},
        "device": str(device),
        "num_images": len(rows),
    }
    write_json(paths["metrics"], metrics)
    plot_curves(paths["curves"], history)
    export_embeddings(cfg, model, rows, device, paths["embeddings_csv"], paths["embeddings_parquet"], paths["schema"])
    copy_config(cfg, resolve_path(cfg, "outputs"))
    update_run_metadata(
        cfg,
        learned_output_dir=str(paths["out_dir"]),
        learned_num_images=len(rows),
        learned_embedding_dim=int(cfg["learned"]["model"].get("embedding_dim", 64)),
        learned_best_epoch=best_epoch,
        learned_test_metrics=test_metrics,
    )
    print("Wrote learned outputs under: {}".format(paths["out_dir"]))


if __name__ == "__main__":
    main()
