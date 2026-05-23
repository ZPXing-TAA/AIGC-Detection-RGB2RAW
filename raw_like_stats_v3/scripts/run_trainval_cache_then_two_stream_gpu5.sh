#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data0/xzp/AIGC/raw_like_stats_v3
CONFIG=${PROJECT}/configs/two_stream_resnet_full_aigibench.yaml
LOG_DIR=${PROJECT}/outputs/full_aigibench_resnet50/logs
mkdir -p "${LOG_DIR}"
cd "${PROJECT}"

export CUDA_VISIBLE_DEVICES=5
export PYTHONUNBUFFERED=1

echo "[$(date -Is)] Starting v3 train/val RAW-like cache then two-stream training on physical GPU 5"
echo "[$(date -Is)] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

source ~/anaconda3/etc/profile.d/conda.sh

echo "[$(date -Is)] Step 1/4: ensure full manifest"
conda activate cisp
python scripts/generate_full_manifest.py --config "${CONFIG}" 2>&1 | tee -a "${LOG_DIR}/manifest.log"

echo "[$(date -Is)] Step 2/4: generate train RAW-like cache"
python scripts/generate_raw_like_cache.py --config "${CONFIG}" --split train --device cuda:0 --batch-size 8 2>&1 | tee -a "${LOG_DIR}/raw_cache_train.log"

echo "[$(date -Is)] Step 3/4: generate val RAW-like cache"
python scripts/generate_raw_like_cache.py --config "${CONFIG}" --split val --device cuda:0 --batch-size 8 2>&1 | tee -a "${LOG_DIR}/raw_cache_val.log"

echo "[$(date -Is)] Step 4/4: train two_stream_resnet50"
conda activate MAS
python scripts/train_two_stream_resnet.py --config "${CONFIG}" --model-mode two_stream_resnet50 --device cuda:0 2>&1 | tee -a "${LOG_DIR}/train_two_stream.log"

echo "[$(date -Is)] Finished v3 train/val cache + two-stream training"
