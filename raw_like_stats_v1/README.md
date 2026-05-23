# raw_like_stats_v1

v1 implements RAW-like statistical probes plus a tabular MLP predictor.

The project reuses RAW-like `.npy` arrays already produced by v0 AIGIBench lite. It does not rerun CycleISP by default, does not train an image-side feature extractor, and does not feed path/source/generator metadata into the MLP.

Default remote target:

```bash
/data1/xzp/AIGC/raw_like_stats_v1
```

Main commands on `ss420h`:

```bash
cd /data1/xzp/AIGC/raw_like_stats_v1
conda activate cisp

python scripts/extract_raw_like_statistical_probes.py --config configs/aigibench_lite_v1.yaml
python scripts/train_mlp_probe.py --config configs/aigibench_lite_v1.yaml
python scripts/analyze_raw_like_statistical_probes.py --config configs/aigibench_lite_v1.yaml
```

End-to-end wrapper:

```bash
bash scripts/run_aigibench_lite_v1.sh --full
```

Smoke run:

```bash
bash scripts/run_aigibench_lite_v1.sh --smoke --epochs 1
```

