#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data0/xzp/AIGC/raw_like_stats_v3
CONFIG=${PROJECT}/configs/two_stream_resnet_full_aigibench.yaml
LOG_DIR=${PROJECT}/outputs/full_aigibench_resnet50/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}"
export PYTHONUNBUFFERED=1

source ~/anaconda3/etc/profile.d/conda.sh

echo "[$(date -Is)] Starting sharded train/val RAW-like cache on physical GPUs 5,7 then two-stream training on GPU 5"

echo "[$(date -Is)] Step 1/4: ensure full manifest"
conda activate cisp
python scripts/generate_full_manifest.py --config "${CONFIG}" 2>&1 | tee -a "${LOG_DIR}/manifest.log"

run_cache_shards() {
  local split=$1
  echo "[$(date -Is)] Generating ${split} RAW-like cache with 2 shards"
  (
    export CUDA_VISIBLE_DEVICES=5
    python scripts/generate_raw_like_cache.py --config "${CONFIG}" --split "${split}" --device cuda:0 --batch-size 8 --num-shards 2 --shard-index 0 2>&1 | tee -a "${LOG_DIR}/raw_cache_${split}_shard0of2.log"
  ) &
  local pid0=$!
  (
    export CUDA_VISIBLE_DEVICES=7
    python scripts/generate_raw_like_cache.py --config "${CONFIG}" --split "${split}" --device cuda:0 --batch-size 8 --num-shards 2 --shard-index 1 2>&1 | tee -a "${LOG_DIR}/raw_cache_${split}_shard1of2.log"
  ) &
  local pid1=$!
  local failed=0
  wait "$pid0" || failed=1
  wait "$pid1" || failed=1
  if [[ "$failed" -ne 0 ]]; then
    echo "[$(date -Is)] ERROR: ${split} cache shard failed" >&2
    exit 1
  fi
}

echo "[$(date -Is)] Step 2/4: generate train RAW-like cache"
run_cache_shards train

echo "[$(date -Is)] Step 3/4: generate val RAW-like cache"
run_cache_shards val

echo "[$(date -Is)] Step 4/4: train two_stream_resnet50 on physical GPU 5"
conda activate MAS
export CUDA_VISIBLE_DEVICES=5
python scripts/train_two_stream_resnet.py --config "${CONFIG}" --model-mode two_stream_resnet50 --device cuda:0 2>&1 | tee -a "${LOG_DIR}/train_two_stream.log"

echo "[$(date -Is)] Finished v3 train/val cache + two-stream training"
