# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

AIGC detection research framework. Core hypothesis: CycleISP RGB→RAW-like conversion exposes traces that help distinguish real photos from AI-generated images. Four progressive versions (v0–v3), each in `raw_like_stats_v*/`.

## Data paths

- **Primary**: `/data0/xzp/AIGC` (project), `/data0/xzp/dataset` (all datasets)
- **Legacy**: `/data1/xzp/AIGC`, `/data1/xzp/dataset` (symlinked to data0 after migration)
- **RAW-RGB pairs**: `/data0/xzp/dataset/rgb-raw/` (NTIRE2025-rgb2raw, MIT-Adobe-FiveK)
- **In-the-wild**: `/data0/xzp/dataset/in-the-wild/` (AIGIBench, BFree-Online, RRDataset, etc.)
- **CycleISP model**: `/data0/xzp/AIGC/CycleISP`

Many scripts reference `/data1/xzp/` — these paths resolve via symlink to `/data0/xzp/`. When writing new code, use `/data0/xzp/`.

## Conda environment

All pipelines use the `cisp` environment. Scripts embed full python paths like `/home/xingzhengpeng/anaconda3/envs/cisp/bin/python` — adjust if the environment name or path differs on the current machine.

## Version architectures

Each version follows: **manifest → feature extraction → train/eval → analysis**.

### v0 — Handcrafted RAW-like stats (233-dim)
- `CycleISP` converts RGB → 4-channel RAW-like `.npy` arrays
- 4 feature groups: noise (36), highpass (146), channel (31), frequency (20)
- Baselines: LR/SVM/RF probes, GMM anomaly detection, small CNN learned extractor
- Pipeline: `raw_like_stats_v0/scripts/run_all_v0.sh`

### v1 — Statistical probes + MLP (817-dim)
- Reuses v0's RAW-like `.npy` arrays (no re-conversion needed)
- 4 feature groups: noise (180), SRM (544), GLCM (70), frequency (23)
- Pipeline: `raw_like_stats_v1/scripts/run_aigibench_lite_v1.sh`
- Config: `configs/aigibench_lite_v1.yaml`

### v2 — Photo-only anomaly detection (compact features)
- Trains only on real photos; generated samples never seen during training
- Compact features: noise20 + AI azimuthal-integral spectrum + quantized SRM stats
- Multiple anomaly detectors: diagonal/full Gaussian, PCA+GMM, kNN, IsolationForest, OneClassSVM
- Pipeline: `raw_like_stats_v2/scripts/run_photo_only_v2.sh`
- Config: `configs/photo_only_aigibench_lite_v2.yaml`

### v3 — Two-stream ResNet50 (end-to-end)
- Three modes: `rgb_only_resnet50`, `raw_only_resnet50`, `two_stream_resnet50`
- RAW-like cache format: `.npy`, `uint16`, `[4,224,224]`, channel order `[R,G1,G2,B]`
- Pipeline: `raw_like_stats_v3/scripts/run_full_aigibench_resnet.sh`
- Config: `configs/two_stream_resnet_full_aigibench.yaml`
- CycleISP cache gen uses `cisp` env; ResNet training uses `MAS` env

## Common commands

```bash
# Download datasets
bash download_scripts.sh
bash download/download_rgb_raw.sh

# Post-process archives (extract zips/tars, generate manifest)
FORCE=1 bash post_processing.sh

# v0 full pipeline
bash raw_like_stats_v0/scripts/run_all_v0.sh --config raw_like_stats_v0/configs/default.yaml

# v1 lite pipeline (smoke test)
bash raw_like_stats_v1/scripts/run_aigibench_lite_v1.sh --smoke

# v1 lite pipeline (full)
bash raw_like_stats_v1/scripts/run_aigibench_lite_v1.sh

# v2 photo-only pipeline (smoke test)
bash raw_like_stats_v2/scripts/run_photo_only_v2.sh --smoke

# v3 full ResNet training
bash raw_like_stats_v3/scripts/run_full_aigibench_resnet.sh
```

All pipeline scripts support `--smoke` for fast validation with minimal data, and `--skip-<step>` flags to resume from a specific stage.

## Key patterns

- **Marker files** (`$MARK_DIR/<name>.done`): prevent re-running completed download/post-processing steps. Delete the marker or set `FORCE=1` to re-run.
- **Config-driven**: all pipelines use YAML configs under each version's `configs/` directory.
- **smoke mode convention**: `LIMIT_PER_SPLIT_LABEL` set to 3–5, `EPOCHS` to 1–2, plots disabled.
- **Output base**: `/data0/xzp/AIGC/raw_like_stats_v<N>/outputs/`
- **Design docs**: `communi/` contains 15 markdown files with experiment rationale, results analysis, and implementation plans — consult before making major architectural changes.
- **srm_filter_kernel.py** at repo root defines SRM kernels used by v1/v2 feature extractors.
