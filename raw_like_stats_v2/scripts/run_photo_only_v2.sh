#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/photo_only_aigibench_lite_v2.yaml"
MODE="full"
OUTPUT_DIR=""
OVERWRITE=""
LIMIT_PER_SPLIT_LABEL=""
MAX_MODELS=""
NO_PLOTS=""
SKIP_EXTRACT=0
SKIP_EVAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke)
      MODE="smoke"
      LIMIT_PER_SPLIT_LABEL="${LIMIT_PER_SPLIT_LABEL:-3}"
      MAX_MODELS="${MAX_MODELS:-6}"
      NO_PLOTS="--no-plots"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --limit-per-split-label)
      LIMIT_PER_SPLIT_LABEL="$2"
      shift 2
      ;;
    --max-models)
      MAX_MODELS="$2"
      shift 2
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    --no-plots)
      NO_PLOTS="--no-plots"
      shift
      ;;
    --skip-extract)
      SKIP_EXTRACT=1
      shift
      ;;
    --skip-eval)
      SKIP_EVAL=1
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
    OUTPUT_DIR="/data1/xzp/AIGC/raw_like_stats_v2/outputs/photo_only_aigibench_lite_v2_smoke"
  else
    OUTPUT_DIR="/data1/xzp/AIGC/raw_like_stats_v2/outputs/photo_only_aigibench_lite_v2"
  fi
fi

mkdir -p "${OUTPUT_DIR}"
echo "mode=${MODE} output=${OUTPUT_DIR}"

EXTRACT_ARGS=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}")
if [[ -n "${LIMIT_PER_SPLIT_LABEL}" ]]; then
  EXTRACT_ARGS+=(--limit-per-split-label "${LIMIT_PER_SPLIT_LABEL}")
fi
if [[ -n "${OVERWRITE}" ]]; then
  EXTRACT_ARGS+=("${OVERWRITE}")
fi

EVAL_ARGS=(--config "${CONFIG}" --output-dir "${OUTPUT_DIR}" --features "${OUTPUT_DIR}/features.parquet")
if [[ -n "${MAX_MODELS}" ]]; then
  EVAL_ARGS+=(--max-models "${MAX_MODELS}")
fi
if [[ -n "${NO_PLOTS}" ]]; then
  EVAL_ARGS+=("${NO_PLOTS}")
fi

if [[ "${SKIP_EXTRACT}" == "0" ]]; then
  python "${ROOT_DIR}/scripts/extract_photo_only_compact_features.py" "${EXTRACT_ARGS[@]}"
fi

if [[ "${SKIP_EVAL}" == "0" ]]; then
  python "${ROOT_DIR}/scripts/evaluate_photo_only_anomaly.py" "${EVAL_ARGS[@]}"
fi

echo "Done: ${OUTPUT_DIR}"

