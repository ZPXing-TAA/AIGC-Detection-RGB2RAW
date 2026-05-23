#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.aigibench_dataset import AIGIBenchRgbRawDataset
from src.models import build_model
from src.training.metrics import compute_detection_metrics
from src.utils.config import ensure_dir, env_snapshot, load_config, save_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RGB/RAW/two-stream ResNet50 on AIGIBench.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--model-mode",
        default=None,
        choices=["rgb_only_resnet50", "raw_only_resnet50", "two_stream_resnet50"],
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit-train", type=int, default=None)
    parser.add_argument("--limit-val", type=int, default=None)
    parser.add_argument("--limit-test", type=int, default=None)
    parser.add_argument("--eval-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override train.batch_size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override data.num_workers.")
    parser.add_argument("--output-dir", default=None, help="Override project.output_dir.")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable ImageNet pretrained weights for smoke tests.")
    return parser.parse_args()


def build_dataset(cfg: Dict[str, Any], split: str, model_mode: str, limit: int = None) -> AIGIBenchRgbRawDataset:
    use_rgb = model_mode in ("rgb_only_resnet50", "two_stream_resnet50")
    use_raw = model_mode in ("raw_only_resnet50", "two_stream_resnet50")
    data_cfg = cfg["data"]
    return AIGIBenchRgbRawDataset(
        manifest_path=data_cfg["manifest_path"],
        split=split,
        image_size=int(data_cfg["image_size"]),
        rgb_mean=data_cfg["rgb_mean"],
        rgb_std=data_cfg["rgb_std"],
        raw_mean=data_cfg["raw_mean"],
        raw_std=data_cfg["raw_std"],
        raw_size=int(data_cfg.get("raw_cache_size", data_cfg["image_size"])),
        use_rgb=use_rgb,
        use_raw=use_raw,
        limit=limit,
    )


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def forward_model(model: nn.Module, batch: Dict[str, Any]) -> torch.Tensor:
    return model(rgb=batch.get("rgb"), raw_like=batch.get("raw_like"))


def softmax_fake(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)[:, 1]


def make_optimizer(model: nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    train_cfg = cfg["train"]
    encoder_lr = float(train_cfg.get("encoder_lr", train_cfg.get("lr", 1e-4)))
    classifier_lr = float(train_cfg.get("classifier_lr", train_cfg.get("lr", 1e-4)))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    params = [
        {"params": list(model.encoder_parameters()), "lr": encoder_lr},
        {"params": list(model.classifier_parameters()), "lr": classifier_lr},
    ]
    optim_cls = getattr(torch.optim, "AdamW", None)
    if optim_cls is None:
        print("torch.optim.AdamW is unavailable in this PyTorch; falling back to Adam.")
        optim_cls = torch.optim.Adam
    return optim_cls(params, lr=float(train_cfg.get("lr", 1e-4)), weight_decay=weight_decay)


def make_scheduler(optimizer: torch.optim.Optimizer, cfg: Dict[str, Any]):
    scheduler_name = str(cfg["train"].get("scheduler", "cosine")).lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "reduce_on_plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(cfg["train"]["epochs"])))


def has_amp() -> bool:
    return hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler")


def autocast_context(enabled: bool):
    if enabled and has_amp():
        return torch.cuda.amp.autocast()
    class Dummy(object):
        def __enter__(self):
            return None
        def __exit__(self, exc_type, exc_val, exc_tb):
            return False
    return Dummy()


def train_one_epoch(model, loader, criterion, optimizer, device, amp_enabled: bool, scaler):
    model.train()
    total_loss = 0.0
    n = 0
    pbar = tqdm(loader, desc="train", leave=False)
    for batch in pbar:
        batch = move_batch_to_device(batch, device)
        labels = batch["label"]
        optimizer.zero_grad()
        with autocast_context(amp_enabled):
            logits = forward_model(model, batch)
            loss = criterion(logits, labels)
        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        bs = int(labels.size(0))
        total_loss += float(loss.detach().cpu().item()) * bs
        n += bs
        pbar.set_postfix(loss=total_loss / max(1, n))
    return total_loss / max(1, n)


def evaluate(model, loader, criterion, device, split: str) -> Dict[str, Any]:
    model.eval()
    losses = []
    labels_all: List[int] = []
    scores_all: List[float] = []
    rows: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=split, leave=False):
            batch = move_batch_to_device(batch, device)
            labels = batch["label"]
            logits = forward_model(model, batch)
            loss = criterion(logits, labels)
            scores = softmax_fake(logits).detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy().astype(int)
            preds = (scores >= 0.5).astype(int)
            losses.append(float(loss.detach().cpu().item()) * int(labels.size(0)))
            labels_all.extend(labels_np.tolist())
            scores_all.extend(scores.tolist())
            batch_size = len(labels_np)
            for i in range(batch_size):
                rows.append(
                    {
                        "image_id": batch["image_id"][i],
                        "dataset": batch["dataset"][i],
                        "split": batch["split"][i],
                        "subset": batch["subset"][i],
                        "generator": batch["generator"][i],
                        "generator_family": batch["generator_family"][i],
                        "source": batch["source"][i],
                        "path": batch["path"][i],
                        "raw_like_path": batch["raw_like_path"][i],
                        "label": int(labels_np[i]),
                        "score_fake": float(scores[i]),
                        "pred_0p5": int(preds[i]),
                    }
                )
    metrics = compute_detection_metrics(
        labels_all,
        scores_all,
        generators=[r["generator"] for r in rows],
    )
    metrics["loss"] = float(np.sum(losses) / max(1, len(labels_all)))
    return {"metrics": metrics, "predictions": pd.DataFrame(rows)}


def save_checkpoint(path: Path, model, optimizer, epoch: int, best_metric: float, cfg: Dict[str, Any], model_mode: str) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "epoch": int(epoch),
            "model_mode": model_mode,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": float(best_metric),
            "config": cfg,
        },
        str(path),
    )


def save_per_generator(metrics: Dict[str, Any], path: Path) -> None:
    rows = metrics.get("per_generator_tpr", [])
    pd.DataFrame(rows).to_csv(str(path), index=False)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = int(args.epochs)
    if args.batch_size is not None:
        cfg["train"]["batch_size"] = int(args.batch_size)
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = int(args.num_workers)
    if args.output_dir is not None:
        cfg["project"]["output_dir"] = str(args.output_dir)
    if args.no_pretrained:
        cfg["model"]["pretrained"] = False
    set_seed(int(cfg["project"].get("seed", 42)))
    model_mode = args.model_mode or cfg["model"]["name"]
    cfg["model"]["name"] = model_mode

    if args.device:
        device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["project"]["output_dir"]) / model_mode
    ensure_dir(out_dir)
    save_config(cfg, str(out_dir / "config.yaml"))
    save_json(env_snapshot(), str(out_dir / "environment.json"))

    train_ds = build_dataset(cfg, "train", model_mode, args.limit_train)
    val_ds = build_dataset(cfg, "val", model_mode, args.limit_val)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["data"].get("num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )

    model = build_model(
        model_mode,
        num_classes=int(cfg["model"].get("num_classes", 2)),
        pretrained=bool(cfg["model"].get("pretrained", True)),
        dropout=float(cfg["model"].get("dropout", 0.3)),
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)
    scheduler = make_scheduler(optimizer, cfg)
    amp_enabled = bool(cfg["train"].get("mixed_precision", True)) and torch.cuda.is_available() and has_amp()
    scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

    best_metric = -1.0
    best_epoch = -1
    stale = 0
    history: List[Dict[str, Any]] = []
    epochs = int(cfg["train"]["epochs"])
    patience = int(cfg["train"].get("patience", 5))
    freeze_backbone_epochs = int(cfg["train"].get("freeze_backbone_epochs", 0))

    for epoch in range(1, epochs + 1):
        model.set_backbone_trainable(epoch > freeze_backbone_epochs)
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, amp_enabled, scaler)
        val_result = evaluate(model, val_loader, criterion, device, "val")
        val_metrics = val_result["metrics"]
        val_auc = float(val_metrics.get("AUROC", float("nan")))
        if np.isnan(val_auc):
            val_auc = -1.0
        row = {"epoch": epoch, "train_loss": float(train_loss)}
        row.update({"val_" + k: v for k, v in val_metrics.items() if not isinstance(v, list)})
        history.append(row)
        pd.DataFrame(history).to_csv(str(out_dir / "training_history.csv"), index=False)
        val_result["predictions"].to_csv(str(out_dir / "predictions_val.csv"), index=False)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_auc)
            else:
                scheduler.step()

        save_checkpoint(out_dir / "last_model.pt", model, optimizer, epoch, best_metric, cfg, model_mode)
        if val_auc > best_metric:
            best_metric = val_auc
            best_epoch = epoch
            stale = 0
            save_checkpoint(out_dir / "best_model.pt", model, optimizer, epoch, best_metric, cfg, model_mode)
        else:
            stale += 1
        print("epoch={} train_loss={:.5f} val_AUROC={:.5f} best={:.5f}@{}".format(epoch, train_loss, val_auc, best_metric, best_epoch))
        if stale >= patience:
            print("Early stopping after {} stale epochs.".format(stale))
            break

    metrics_payload: Dict[str, Any] = {
        "model_mode": model_mode,
        "best_epoch": int(best_epoch),
        "best_val_AUROC": float(best_metric),
        "history": history,
    }

    best_ckpt = torch.load(str(out_dir / "best_model.pt"), map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    val_result = evaluate(model, val_loader, criterion, device, "val")
    val_result["predictions"].to_csv(str(out_dir / "predictions_val.csv"), index=False)
    save_per_generator(val_result["metrics"], out_dir / "per_generator_val.csv")
    metrics_payload["val"] = val_result["metrics"]

    eval_test = bool(args.eval_test or cfg["train"].get("evaluate_test", False))
    if eval_test:
        test_ds = build_dataset(cfg, "test", model_mode, args.limit_test)
        test_loader = DataLoader(
            test_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=int(cfg["data"].get("num_workers", 4)),
            pin_memory=torch.cuda.is_available(),
        )
        test_result = evaluate(model, test_loader, criterion, device, "test")
        test_result["predictions"].to_csv(str(out_dir / "predictions_test.csv"), index=False)
        save_per_generator(test_result["metrics"], out_dir / "per_generator_test.csv")
        metrics_payload["test"] = test_result["metrics"]
    else:
        empty_cols = [
            "image_id",
            "dataset",
            "split",
            "subset",
            "generator",
            "generator_family",
            "source",
            "path",
            "raw_like_path",
            "label",
            "score_fake",
            "pred_0p5",
        ]
        pd.DataFrame(columns=empty_cols).to_csv(str(out_dir / "predictions_test.csv"), index=False)

    with open(str(out_dir / "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2, sort_keys=True)
    print("Saved outputs under {}".format(out_dir))


if __name__ == "__main__":
    main()
