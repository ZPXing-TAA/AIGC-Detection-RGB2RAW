#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$PWD/aigc_in_the_wild_benchmarks}"
LOG_DIR="$ROOT/_logs"

mkdir -p "$ROOT" "$LOG_DIR"

log() {
  printf "\n[%s] %s\n" "$(date '+%F %T')" "$*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1"
    echo "Install deps first, e.g.: python -m pip install -U huggingface_hub hf_transfer hf_xet gdown zenodo_get"
    exit 1
  }
}

run_step() {
  local name="$1"
  shift
  log "START $name"
  {
    echo "===== $name ====="
    echo "cwd: $ROOT"
    echo "cmd: $*"
    echo
    "$@"
  } 2>&1 | tee "$LOG_DIR/${name}.log"
  log "DONE  $name"
}

cd "$ROOT"

need_cmd gdown
need_cmd hf
need_cmd zenodo_get
need_cmd wget
need_cmd git

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

run_step "01_chameleon" \
  gdown --fuzzy -O Chameleon.zip \
  "https://drive.google.com/file/d/1QLYJMhy0CbBVT01BLkkw7KPPL5BpmxnH/view"

run_step "02_synthwildx" \
  hf download nebula/DF-arrow \
  --repo-type dataset \
  --include "synthwildx/**" \
  --local-dir synthwildx_hf

run_step "03_wildrf" \
  gdown --fuzzy -O WildRF.zip \
  "https://drive.google.com/file/d/1A0xoL44Yg68ixd-FuIJn2VC4vdZ6M2gn/view?usp=sharing"

run_step "04_aigibench" \
  hf download HorizonTEL/AIGIBench \
  --repo-type dataset \
  --local-dir AIGIBench

run_step "05_cospy_bench" \
  hf download ruojiruoli/Co-Spy-Bench \
  --repo-type dataset \
  --local-dir Co-Spy-Bench

run_step "06_rrdataset" \
  zenodo_get -r 14963880 -o RRDataset

if [ ! -d "B-Free-repo/.git" ]; then
  run_step "07_bfree_repo" \
    git clone https://github.com/grip-unina/B-Free.git B-Free-repo
else
  run_step "07_bfree_repo" \
    git -C B-Free-repo pull --ff-only
fi

run_step "08_bfree_download_index" \
  wget -c -r -np -nH --cut-dirs=3 \
  -P B-Free-downloads \
  "https://www.grip.unina.it/download/prog/B-Free/"

log "All downloads finished under $ROOT"
