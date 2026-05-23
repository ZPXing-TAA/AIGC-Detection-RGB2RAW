#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$ROOT/configs/default.yaml"
PYTHON_BIN="${PYTHON_BIN:-/home/xingzhengpeng/anaconda3/envs/cisp/bin/python}"
OVERWRITE_ARGS=()
LIMIT_ARGS=()
TRAIN_LEARNED=0
LEARNED_EPOCH_ARGS=()

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export NUMBA_NUM_THREADS="${NUMBA_NUM_THREADS:-4}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE_ARGS=(--overwrite)
      shift
      ;;
    --limit)
      LIMIT_ARGS=(--limit "$2")
      shift 2
      ;;
    --train-learned)
      TRAIN_LEARNED=1
      shift
      ;;
    --learned-epochs)
      LEARNED_EPOCH_ARGS=(--epochs "$2")
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"

"$PYTHON_BIN" scripts/prepare_dataset.py --config "$CONFIG" "${OVERWRITE_ARGS[@]}"
"$PYTHON_BIN" scripts/run_cycleisp_rgb2raw.py --config "$CONFIG" --split photo "${OVERWRITE_ARGS[@]}" "${LIMIT_ARGS[@]}"
"$PYTHON_BIN" scripts/run_cycleisp_rgb2raw.py --config "$CONFIG" --split gen "${OVERWRITE_ARGS[@]}" "${LIMIT_ARGS[@]}"
"$PYTHON_BIN" scripts/extract_raw_like_stats.py --config "$CONFIG" "${OVERWRITE_ARGS[@]}" "${LIMIT_ARGS[@]}"
"$PYTHON_BIN" scripts/analyze_raw_like_stats.py --config "$CONFIG" --features outputs/features/raw_like_features.parquet "${OVERWRITE_ARGS[@]}"
if [ "$TRAIN_LEARNED" -eq 1 ]; then
  "$PYTHON_BIN" scripts/train_learned_extractor.py --config "$CONFIG" "${OVERWRITE_ARGS[@]}" "${LIMIT_ARGS[@]}" "${LEARNED_EPOCH_ARGS[@]}"
fi
