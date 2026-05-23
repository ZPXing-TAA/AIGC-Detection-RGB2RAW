from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(Path(config_path).resolve())
    return cfg


def project_root(cfg: Dict[str, Any]) -> Path:
    root = cfg.get("project", {}).get("root")
    if root:
        return Path(root)
    return Path.cwd()


def resolve_path(cfg: Dict[str, Any], path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return project_root(cfg) / path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_config(cfg: Dict[str, Any], output_dir: Path) -> None:
    ensure_dir(output_dir)
    config_path = cfg.get("_config_path")
    if config_path and Path(config_path).exists():
        shutil.copy2(config_path, output_dir / "config_used.yaml")


def cycleisp_git_commit(repo_dir: str) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", repo_dir, "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def update_run_metadata(cfg: Dict[str, Any], **updates: Any) -> None:
    root = project_root(cfg)
    path = root / "outputs" / "run_metadata.json"
    payload = read_json(path)
    payload.update(
        {
            "project_root": str(root),
            "feature_config": cfg.get("_config_path", ""),
            "cycleisp_repo": cfg["cycleisp"]["repo_dir"],
            "cycleisp_checkpoint": cfg["cycleisp"]["checkpoint_path"],
            "cycleisp_commit": cycleisp_git_commit(cfg["cycleisp"]["repo_dir"]),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    if "created_at" not in payload:
        payload["created_at"] = payload["updated_at"]
    payload.update(updates)
    write_json(path, payload)
