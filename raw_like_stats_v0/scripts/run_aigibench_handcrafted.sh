#!/usr/bin/env bash
set -euo pipefail

CONFIG="configs/aigibench_handcrafted.yaml"
MODE="smoke"
OVERWRITE=""
LEGACY_LIMIT="2"
TRAIN_LIMIT=""
VAL_LIMIT=""
TEST_LIMIT=""
BATCH_SIZE=""
DEVICE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG="$2"
      shift 2
      ;;
    --smoke)
      MODE="smoke"
      shift
      ;;
    --mini)
      MODE="mini"
      shift
      ;;
    --lite)
      MODE="lite"
      shift
      ;;
    --full)
      MODE="full"
      shift
      ;;
    --overwrite)
      OVERWRITE="--overwrite"
      shift
      ;;
    --max-per-label-per-subset)
      LEGACY_LIMIT="$2"
      TRAIN_LIMIT="$2"
      VAL_LIMIT="$2"
      TEST_LIMIT="$2"
      shift 2
      ;;
    --train-per-label-per-subset)
      TRAIN_LIMIT="$2"
      shift 2
      ;;
    --val-per-label-per-subset)
      VAL_LIMIT="$2"
      shift 2
      ;;
    --test-per-label-per-subset)
      TEST_LIMIT="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$MODE" == "smoke" ]]; then
  TRAIN_LIMIT="${TRAIN_LIMIT:-$LEGACY_LIMIT}"
  VAL_LIMIT="${VAL_LIMIT:-$LEGACY_LIMIT}"
  TEST_LIMIT="${TEST_LIMIT:-$LEGACY_LIMIT}"
elif [[ "$MODE" == "mini" ]]; then
  TRAIN_LIMIT="${TRAIN_LIMIT:-100}"
  VAL_LIMIT="${VAL_LIMIT:-100}"
  TEST_LIMIT="${TEST_LIMIT:-100}"
elif [[ "$MODE" == "lite" ]]; then
  TRAIN_LIMIT="${TRAIN_LIMIT:-500}"
  VAL_LIMIT="${VAL_LIMIT:-200}"
  TEST_LIMIT="${TEST_LIMIT:-500}"
elif [[ "$MODE" == "full" ]]; then
  if [[ -n "$TRAIN_LIMIT" || -n "$VAL_LIMIT" || -n "$TEST_LIMIT" ]]; then
    MODE="sampled"
  fi
else
  echo "Unknown mode: $MODE" >&2
  exit 2
fi

RUN_CONFIG="$CONFIG"
OUTPUT_NAME="aigibench_handcrafted"
if [[ "$MODE" != "full" ]]; then
  OUTPUT_NAME="aigibench_handcrafted_${MODE}"
  RUN_CONFIG="$(mktemp "/tmp/${OUTPUT_NAME}.XXXXXX.yaml")"
  python - "$CONFIG" "$RUN_CONFIG" "$OUTPUT_NAME" <<'PY'
import sys
import yaml

src, dst, output_name = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["data"]["manifest_path"] = "data/manifests/{}.csv".format(output_name)
cfg["data"]["manifest_summary_path"] = "outputs/{}/manifest_summary.json".format(output_name)
cfg["data"]["failed_images_path"] = "outputs/{}/failed_images.csv".format(output_name)
cfg["raw_like_output"]["photo_dir"] = "outputs/{}/raw_like/photo".format(output_name)
cfg["raw_like_output"]["gen_dir"] = "outputs/{}/raw_like/gen".format(output_name)
cfg["raw_like_output"]["preview_dir"] = "outputs/{}/raw_like/previews".format(output_name)
cfg["raw_like_output"]["save_preview_png"] = False
cfg["features"]["output_csv"] = "outputs/{}/features/raw_like_features.csv".format(output_name)
cfg["features"]["output_parquet"] = "outputs/{}/features/raw_like_features.parquet".format(output_name)
cfg["features"]["schema_json"] = "outputs/{}/features/feature_schema.json".format(output_name)
cfg["analysis"]["output_dir"] = "outputs/{}/analysis".format(output_name)
cfg["aigibench_eval"]["output_dir"] = "outputs/{}/evaluation".format(output_name)
with open(dst, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PY
fi

FEATURES_PATH="outputs/${OUTPUT_NAME}/features/raw_like_features.parquet"
PREPARE_ARGS=(--dataset aigibench --config "$RUN_CONFIG")
RUN_ARGS=(--config "$RUN_CONFIG" --split all)
EXTRACT_ARGS=(--config "$RUN_CONFIG")
EVAL_ARGS=(--config "$RUN_CONFIG" --features "$FEATURES_PATH")

if [[ -n "$OVERWRITE" ]]; then
  PREPARE_ARGS+=("$OVERWRITE")
  RUN_ARGS+=("$OVERWRITE")
  EXTRACT_ARGS+=("$OVERWRITE")
  EVAL_ARGS+=("$OVERWRITE")
fi
if [[ -n "$TRAIN_LIMIT" ]]; then
  PREPARE_ARGS+=(--max-train-per-label-per-subset "$TRAIN_LIMIT")
fi
if [[ -n "$VAL_LIMIT" ]]; then
  PREPARE_ARGS+=(--max-val-per-label-per-subset "$VAL_LIMIT")
fi
if [[ -n "$TEST_LIMIT" ]]; then
  PREPARE_ARGS+=(--max-test-per-label-per-subset "$TEST_LIMIT")
fi
if [[ -n "$BATCH_SIZE" ]]; then
  RUN_ARGS+=(--batch-size "$BATCH_SIZE")
fi
if [[ -n "$DEVICE" ]]; then
  RUN_ARGS+=(--device "$DEVICE")
fi
if [[ "$MODE" == "smoke" ]]; then
  EVAL_ARGS+=(--smoke)
fi

echo "mode=${MODE} output=outputs/${OUTPUT_NAME}"
echo "limits train=${TRAIN_LIMIT:-full} val=${VAL_LIMIT:-full} test=${TEST_LIMIT:-full}"
python scripts/prepare_dataset.py "${PREPARE_ARGS[@]}"
python scripts/run_cycleisp_rgb2raw.py "${RUN_ARGS[@]}"
python scripts/extract_raw_like_stats.py "${EXTRACT_ARGS[@]}"
python scripts/evaluate_aigibench_handcrafted.py "${EVAL_ARGS[@]}"
