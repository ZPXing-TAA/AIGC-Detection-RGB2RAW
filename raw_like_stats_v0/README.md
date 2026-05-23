# RAW-like AIGC Statistics v0

This framework runs a first-stage RAW-like statistics feasibility pipeline:

1. Build a balanced BFree REAL/FAKE manifest.
2. Convert sRGB images to CycleISP RAW-like packed RGGB arrays.
3. Extract handcrafted RAW-like statistics.
4. Analyze class separability with embeddings, histograms, GMM likelihood, a linear probe, and feature ablations.

Default remote paths assume:

- Project: `/data1/xzp/AIGC/raw_like_stats_v0`
- CycleISP: `/data1/xzp/AIGC/CycleISP`
- BFree images: `/data1/xzp/dataset/in-the-wild/BFree-Online/images`
- Python: `/home/xingzhengpeng/anaconda3/envs/cisp/bin/python`

Run:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v0
bash scripts/run_all_v0.sh
```

Debug run:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v0
bash scripts/run_all_v0.sh --overwrite --limit 2
```

Expected main outputs:

- `outputs/raw_like/photo/*.npy`
- `outputs/raw_like/gen/*.npy`
- `outputs/features/raw_like_features.csv`
- `outputs/features/raw_like_features.parquet`
- `outputs/features/feature_schema.json`
- `outputs/analysis/umap_raw_like.png`
- `outputs/analysis/tsne_raw_like.png`
- `outputs/analysis/gmm_likelihood_hist.png`
- `outputs/analysis/linear_probe_results.json`
- `outputs/analysis/feature_ablation_results.json`

Optional learned extractor training:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v0
bash scripts/run_all_v0.sh --train-learned
```

Debug learned training:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v0
/home/xingzhengpeng/anaconda3/envs/cisp/bin/python scripts/train_learned_extractor.py \
  --config configs/default.yaml \
  --overwrite \
  --limit 8 \
  --epochs 1
```

Learned outputs:

- `outputs/learned/checkpoints/best.pt`
- `outputs/learned/learned_metrics.json`
- `outputs/learned/learned_embeddings.csv`
- `outputs/learned/learned_embeddings.parquet`
- `outputs/learned/learned_embedding_schema.json`

Analyze learned embeddings:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v0
/home/xingzhengpeng/anaconda3/envs/cisp/bin/python scripts/analyze_learned_features.py \
  --config configs/default.yaml \
  --overwrite
```

Learned analysis outputs:

- `outputs/learned_analysis/umap_learned.png`
- `outputs/learned_analysis/tsne_learned.png`
- `outputs/learned_analysis/gmm_likelihood_hist.png`
- `outputs/learned_analysis/gmm_likelihood_results.json`

## AIGIBench Handcrafted Evaluation

This protocol uses only handcrafted RAW-like statistics. It does not train or use the learned
feature extractor. AIGIBench must be unpacked under `/data1/xzp/dataset/in-the-wild/AIGIBench`.

Smoke run, for pipeline validation only:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v0
bash scripts/run_aigibench_handcrafted.sh --smoke --overwrite
```

Smoke outputs are isolated under `outputs/aigibench_handcrafted_smoke/` and
`data/manifests/aigibench_handcrafted_smoke.csv`, so they do not replace the full-run manifest.

Sampled runs:

```bash
# Around 100 images per split/subset/label bucket.
bash scripts/run_aigibench_handcrafted.sh --mini --overwrite --batch-size 8

# Around train=500, val=200, test=500 per subset/label bucket.
bash scripts/run_aigibench_handcrafted.sh --lite --overwrite --batch-size 8

# Custom sampled protocol.
bash scripts/run_aigibench_handcrafted.sh --lite --overwrite \
  --train-per-label-per-subset 300 \
  --val-per-label-per-subset 150 \
  --test-per-label-per-subset 300 \
  --batch-size 8
```

The AIGIBench handcrafted config disables preview PNGs by default to avoid extra IO.

Full run:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v0
bash scripts/run_aigibench_handcrafted.sh --full
```

Evaluation protocol:

- Logistic Regression fits `SimpleImputer + StandardScaler` on train features only, trains on scaled train features, selects one global threshold on pooled val, then evaluates frozen parameters on test.
- GMM fits `SimpleImputer + StandardScaler + PCA + GaussianMixture(reg_covar=1e-5)` on train real/photo features only, selects one global anomaly threshold on pooled val, then evaluates frozen parameters on test.
- Per-generator test metrics use each generator subset's `0_real` images as negatives and `1_fake` images as positives. Smoke metrics are not experimental conclusions.

Main AIGIBench outputs:

- `outputs/aigibench_handcrafted/manifest_summary.json`
- `outputs/aigibench_handcrafted/failed_images.csv`
- `outputs/aigibench_handcrafted/evaluation/probe_per_generator_metrics.csv`
- `outputs/aigibench_handcrafted/evaluation/gmm_per_generator_metrics.csv`
- `outputs/aigibench_handcrafted/evaluation/feature_group_ablation.csv`
- `outputs/aigibench_handcrafted/evaluation/feature_effect_size.csv`
- `outputs/aigibench_handcrafted/evaluation/thresholds.json`
- `outputs/aigibench_handcrafted/evaluation/umap_handcrafted_label.png`
- `outputs/aigibench_handcrafted/evaluation/tsne_handcrafted_label.png`
