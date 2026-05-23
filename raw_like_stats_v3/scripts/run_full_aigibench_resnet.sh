#!/usr/bin/env bash
set -euo pipefail

CONFIG=${CONFIG:-configs/two_stream_resnet_full_aigibench.yaml}
DEVICE=${DEVICE:-cuda:0}
RAW_BATCH_SIZE=${RAW_BATCH_SIZE:-8}
MODEL_MODE=${MODEL_MODE:-two_stream_resnet50}

python scripts/generate_full_manifest.py --config "${CONFIG}"
python scripts/generate_raw_like_cache.py --config "${CONFIG}" --split train --device "${DEVICE}" --batch-size "${RAW_BATCH_SIZE}"
python scripts/generate_raw_like_cache.py --config "${CONFIG}" --split val --device "${DEVICE}" --batch-size "${RAW_BATCH_SIZE}"
python scripts/train_two_stream_resnet.py --config "${CONFIG}" --model-mode "${MODEL_MODE}" --device "${DEVICE}"

