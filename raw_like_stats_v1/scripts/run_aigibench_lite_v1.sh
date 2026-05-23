#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/aigibench_lite_v1.yaml"
MODE="full"
DEVICE="cuda:0"
EPOCHS=""
OVERWRITE=""
OUTPUT_DIR=""
SKIP_EXTRACT=0
SKIP_TRAIN=0
SKIP_ANALYSIS=0
GROUP_ABLATION=1
LIMIT_PER_SPLIT_LABEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      LIMIT_PER_SPLIT_LABEL="${LIMIT_PER_SPLIT_LABEL:-5}"
      EPOCHS="${EPOCHS:-1}"
      GROUP_ABLATION=0
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --limit-per-split-label)
      LIMIT_PER_SPLIT_LABEL="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    --skip-extract)
      SKIP_EXTRACT=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-analysis)
      SKIP_ANALYSIS=1
      shift
      ;;
    --no-group-ablation)
      GROUP_ABLATION=0
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${OUTPUT_DIR}" ]]; then
  if [[ "${MODE}" == "smoke" ]]; then
    OUTPUT_DIR="/data1/xzp/AIGC/raw_like_stats_v1/outputs/aigibench_lite_v1_smoke"
  else
    OUTPUT_DIR="/data1/xzp/AIGC/raw_like_stats_v1/outputs/aigibench_lite_v1"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
echo "mode=${MODE} output=${OUTPUT_DIR} device=${DEVICE}"

EXTRACT_ARGS=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}")
if [[ -n "${LIMIT_PER_SPLIT_LABEL}" ]]; then
  EXTRACT_ARGS+=(--limit-per-split-label "${LIMIT_PER_SPLIT_LABEL}")
fi
if [[ -n "${OVERWRITE}" ]]; then
  EXTRACT_ARGS+=("${OVERWRITE}")
fi

TRAIN_ARGS=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}" --features "${OUTPUT_DIR}/features.parquet" --device "${DEVICE}")
if [[ -n "${EPOCHS}" ]]; then
  TRAIN_ARGS+=(--epochs "${EPOCHS}")
fi
if [[ "${GROUP_ABLATION}" == "0" ]]; then
  TRAIN_ARGS+=(--no-group-ablation)
fi

ANALYSIS_ARGS=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}" --features "${OUTPUT_DIR}/features.parquet")

if [[ "${SKIP_EXTRACT}" == "0" ]]; then
  python "${ROOT_DIR}/scripts/extract_raw_like_statistical_probes.py" "${EXTRACT_ARGS[@]}"
fi

if [[ "${SKIP_TRAIN}" == "0" ]]; then
  python "${ROOT_DIR}/scripts/train_mlp_probe.py" "${TRAIN_ARGS[@]}"
fi

if [[ "${SKIP_ANALYSIS}" == "0" ]]; then
  python "${ROOT_DIR}/scripts/analyze_raw_like_statistical_probes.py" "${ANALYSIS_ARGS[@]}"
fi

echo "Done: ${OUTPUT_DIR}"

