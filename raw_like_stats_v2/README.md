# raw_like_stats_v2

v2 implements photo-only anomaly detection on compact RAW-like statistical features.

Core rule:

```text
Generated/fake samples are never used for training, threshold selection, model selection, feature normalization, PCA fitting, or density fitting.
```

Default remote target:

```bash
/data1/xzp/AIGC/raw_like_stats_v2
```

Main commands on `ss420h`:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v2
conda activate cisp

python scripts/extract_photo_only_compact_features.py --config configs/photo_only_aigibench_lite_v2.yaml
python scripts/evaluate_photo_only_anomaly.py --config configs/photo_only_aigibench_lite_v2.yaml
```

Smoke run:

```bash
bash scripts/run_photo_only_v2.sh --smoke
```

Full run:

```bash
bash scripts/run_photo_only_v2.sh --full --overwrite
```

