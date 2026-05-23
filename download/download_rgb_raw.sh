#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data1/xzp/AIGC"
DATASETS_DIR="/data1/xzp/dataset/rgb-raw"
SCRIPT_DIR="$PROJECT_DIR/download"
LOG_DIR="$SCRIPT_DIR/logs_rgb_raw"
MARK_DIR="$SCRIPT_DIR/.rgb_raw_markers"

mkdir -p "$DATASETS_DIR" "$LOG_DIR" "$MARK_DIR"

log() {
  printf "\n[%s] %s\n" "$(date '+%F %T')" "$*"
}

pick_python() {
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    echo "No python/python3 found" >&2
    exit 1
  fi
}

PYTHON_BIN="${PYTHON_BIN:-$(pick_python)}"
export PATH="$HOME/.local/bin:$PATH"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

install_python_tools() {
  if ! command -v hf >/dev/null 2>&1; then
    log "Installing Hugging Face CLI into user site-packages"
    "$PYTHON_BIN" -m pip install --user -U "huggingface_hub[cli]" hf_transfer hf_xet
    export PATH="$HOME/.local/bin:$PATH"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "Missing command: $1" >&2
    exit 1
  }
}

run_step() {
  local name="$1"
  shift
  local marker="$MARK_DIR/$name.done"

  if [ -f "$marker" ] && [ "${FORCE:-0}" != "1" ]; then
    log "SKIP  $name (marker exists; set FORCE=1 to rerun)"
    return 0
  fi

  log "START $name"
  {
    echo "===== $name ====="
    echo "target: $DATASETS_DIR"
    echo "cmd: $*"
    echo
    "$@"
  } 2>&1 | tee "$LOG_DIR/$name.log"
  touch "$marker"
  log "DONE  $name"
}

download_ntire_rgb2raw() {
  local dest="$DATASETS_DIR/NTIRE2025-rgb2raw"
  mkdir -p "$dest"

  # Official NTIRE 2025 RGB-to-RAW dataset:
  # https://huggingface.co/datasets/marcosv/rgb2raw
  hf download marcosv/rgb2raw \
    train_rgb2raw.zip \
    val_rgbs_in.zip \
    test_rgbs.zip \
    README.md \
    --repo-type dataset \
    --local-dir "$dest"

  cat > "$dest/SOURCE.txt" <<'EOF'
Official source: https://huggingface.co/datasets/marcosv/rgb2raw
Challenge: https://codalab.lisn.upsaclay.fr/competitions/21648
Workshop: https://www.cvlai.net/ntire/2025/
Files expected from official dataset:
- train_rgb2raw.zip
- val_rgbs_in.zip
- test_rgbs.zip
EOF
}

download_mit_adobe_fivek() {
  local dest="$DATASETS_DIR/MIT-Adobe-FiveK"
  local url_base="https://data.csail.mit.edu/graphics/fivek"
  local tar_path="$dest/fivek_dataset.tar"
  local sha_path="$dest/fivek_dataset.tar.sha1"
  mkdir -p "$dest"

  # Official MIT-Adobe FiveK page provides a single archive and SHA1.
  wget -c -O "$tar_path" "$url_base/fivek_dataset.tar"
  wget -c -O "$sha_path" "$url_base/fivek_dataset.tar.sha1"

  if command -v sha1sum >/dev/null 2>&1; then
    local expected
    local actual
    expected="$(awk '{print $NF}' "$sha_path")"
    actual="$(sha1sum "$tar_path" | awk '{print $1}')"
    if [ "$expected" != "$actual" ]; then
      echo "FiveK SHA1 mismatch: expected $expected got $actual" >&2
      exit 2
    fi
  fi

  cat > "$dest/SOURCE.txt" <<'EOF'
Official source: https://data.csail.mit.edu/graphics/fivek/
Official archive: https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar
Official SHA1: https://data.csail.mit.edu/graphics/fivek/fivek_dataset.tar.sha1
The official archive includes DNG inputs, Lightroom catalog/renditions, semantic metadata, and licenses.
EOF
}

install_python_tools
need_cmd curl
need_cmd wget
need_cmd unzip
need_cmd hf

run_step "01_ntire2025_rgb2raw" download_ntire_rgb2raw
run_step "02_mit_adobe_fivek" download_mit_adobe_fivek

log "All RGB/RAW downloads finished under $DATASETS_DIR"
