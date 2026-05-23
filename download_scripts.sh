#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data1/xzp/AIGC"
DATASETS_DIR="/data1/xzp/dataset"
LOG_DIR="$PROJECT_DIR/download_logs"
MARK_DIR="$PROJECT_DIR/.download_markers"

mkdir -p "$DATASETS_DIR" "$LOG_DIR" "$MARK_DIR"

log() {
  printf "\n[%s] %s\n" "$(date '+%F %T')" "$*"
}

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    echo "No python/python3 found" >&2
    exit 1
  fi
}

PYTHON_BIN="${PYTHON_BIN:-$(pick_python)}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

install_python_tools() {
  local missing=0
  command -v hf >/dev/null 2>&1 || missing=1
  command -v zenodo_get >/dev/null 2>&1 || missing=1

  if [ "$missing" -eq 1 ]; then
    log "Installing Python download tools into user site-packages"
    "$PYTHON_BIN" -m pip install --user -U \
      "huggingface_hub[cli]" hf_transfer hf_xet zenodo_get
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1"
    exit 1
  }
}

run_step() {
  local name="$1"
  shift
  local marker="$MARK_DIR/$name.done"

  if [ -f "$marker" ] && [ "${FORCE:-0}" != "1" ]; then
    log "SKIP  $name (marker exists; set FORCE=1 to rerun)"
    return 0
  fi

  log "START $name"
  {
    echo "===== $name ====="
    echo "datasets: $DATASETS_DIR"
    echo "cmd: $*"
    echo
    "$@"
  } 2>&1 | tee "$LOG_DIR/$name.log"
  touch "$marker"
  log "DONE  $name"
}

download_bfree_online() {
  local dest="$DATASETS_DIR/BFree-Online"
  local csv="$dest/BFree_viral_images.csv"
  mkdir -p "$dest/images"

  curl -L \
    "https://raw.githubusercontent.com/grip-unina/B-Free/main/viral_images_dataset/BFree_viral_images.csv" \
    -o "$csv"

  BFREE_CSV="$csv" BFREE_DEST="$dest/images" "$PYTHON_BIN" <<'PY'
import csv
import hashlib
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

csv_path = Path(os.environ["BFREE_CSV"])
dest = Path(os.environ["BFREE_DEST"])
dest.mkdir(parents=True, exist_ok=True)

def md5sum(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def fetch(url, out_path, expected_md5):
    if out_path.exists() and expected_md5 and md5sum(out_path).lower() == expected_md5.lower():
        return "exists"
    if out_path.exists() and not expected_md5:
        return "exists"

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (AIGC benchmark downloader)"},
    )
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as r, tmp.open("wb") as f:
                while True:
                    chunk = r.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
            if expected_md5 and md5sum(tmp).lower() != expected_md5.lower():
                tmp.unlink(missing_ok=True)
                return "md5_failed"
            tmp.replace(out_path)
            return "downloaded"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == 2:
                print(f"FAILED {url}: {exc}", file=sys.stderr)
                return "failed"
            time.sleep(2 + attempt * 3)
    return "failed"

counts = {"downloaded": 0, "exists": 0, "failed": 0, "md5_failed": 0, "skipped": 0}
with csv_path.open(newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        url = (row.get("url") or "").strip()
        filename = (row.get("filename") or "").strip()
        expected_md5 = (row.get("md5") or "").strip()
        if not url or not filename:
            counts["skipped"] += 1
            continue
        out_path = dest / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)
        status = fetch(url, out_path, expected_md5)
        counts[status] = counts.get(status, 0) + 1
        done = counts["downloaded"] + counts["exists"] + counts["failed"] + counts["md5_failed"]
        if done % 50 == 0:
            print(counts, flush=True)

print(counts)
if counts.get("failed", 0) or counts.get("md5_failed", 0):
    sys.exit(2)
PY
}

install_python_tools
need_cmd curl
need_cmd wget
need_cmd hf
need_cmd zenodo_get

cd "$DATASETS_DIR"

run_step "04_aigibench" \
  hf download HorizonTEL/AIGIBench \
  --repo-type dataset \
  --local-dir "$DATASETS_DIR/AIGIBench"

run_step "05_cospy_bench" \
  hf download ruojiruoli/Co-Spy-Bench \
  --repo-type dataset \
  --local-dir "$DATASETS_DIR/Co-Spy-Bench"

run_step "06_rrdataset" \
  zenodo_get -r 14963880 -o "$DATASETS_DIR/RRDataset"

run_step "07_bfree_online" \
  download_bfree_online

log "All downloads finished. Data: $DATASETS_DIR Logs: $LOG_DIR"
