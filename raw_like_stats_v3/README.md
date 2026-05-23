# raw_like_stats_v3

Full-size AIGIBench ResNet50 baselines for AIGC image detection.

This project studies whether RGB-to-RAW-like representations help distinguish
real photographic images from AI-generated images. It is binary classification:

- `0_real` -> label `0`, source `photo`
- `1_fake` -> label `1`, source `gen`
- `1_false` -> label `1`, source `gen`

This is not RGB-vs-RAW classification. Real and fake images both start as RGB
files and use the same RGB-to-RAW-like conversion pipeline.

## Main Commands

Generate the full official AIGIBench manifest:

```bash
cd /data0/xzp/AIGC/raw_like_stats_v3
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cisp
python scripts/generate_full_manifest.py \
  --config configs/two_stream_resnet_full_aigibench.yaml
```

Generate RAW-like cache by split:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cisp

python scripts/generate_raw_like_cache.py \
  --config configs/two_stream_resnet_full_aigibench.yaml \
  --split train --device cuda:0 --batch-size 8

python scripts/generate_raw_like_cache.py \
  --config configs/two_stream_resnet_full_aigibench.yaml \
  --split val --device cuda:0 --batch-size 8

python scripts/generate_raw_like_cache.py \
  --config configs/two_stream_resnet_full_aigibench.yaml \
  --split test --device cuda:0 --batch-size 8
```

Train/evaluate RGB-only:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cisp

python scripts/train_two_stream_resnet.py \
  --config configs/two_stream_resnet_full_aigibench.yaml \
  --model-mode rgb_only_resnet50
```

Train/evaluate RAW-only:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cisp

python scripts/train_two_stream_resnet.py \
  --config configs/two_stream_resnet_full_aigibench.yaml \
  --model-mode raw_only_resnet50
```

Train/evaluate two-stream RGB+RAW:

```bash
source ~/anaconda3/etc/profile.d/conda.sh
conda activate cisp

python scripts/train_two_stream_resnet.py \
  --config configs/two_stream_resnet_full_aigibench.yaml \
  --model-mode two_stream_resnet50
```

The default RAW-like cache is `.npy` with dtype `uint16`, layout `[4,224,224]`,
and channel order `[R,G1,G2,B]`. Values are stored as `round(65535 * x)` for
normalized `x in [0,1]`.
