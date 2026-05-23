#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="/data1/xzp/dataset"
RGB_RAW_ROOT="$DATASET_ROOT/rgb-raw"
IN_THE_WILD_ROOT="$DATASET_ROOT/in-the-wild"
LOG_DIR="$DATASET_ROOT/post_processing_logs"
MARK_DIR="$DATASET_ROOT/.post_processing_markers"

mkdir -p "$LOG_DIR" "$MARK_DIR"

log() {
  printf "\n[%s] %s\n" "$(date '+%F %T')" "$*"
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
  set +e
  {
    echo "===== $name ====="
    echo "root: $DATASET_ROOT"
    echo "cmd: $*"
    echo
    "$@"
  } 2>&1 | tee "$LOG_DIR/$name.log"
  local status="${PIPESTATUS[0]}"
  set -e

  if [ "$status" -eq 10 ]; then
    log "SKIP  $name (required archive not present; marker not written)"
    return 0
  fi
  if [ "$status" -ne 0 ]; then
    log "FAIL  $name (exit $status)"
    return "$status"
  fi

  touch "$marker"
  log "DONE  $name"
}

extract_zip_to_dir() {
  local archive="$1"
  local dest="$2"
  [ -f "$archive" ] || {
    echo "Missing archive, skip: $archive"
    return 0
  }
  mkdir -p "$dest"
  unzip -q -n "$archive" -d "$dest"
}

extract_tar_to_dir() {
  local archive="$1"
  local dest="$2"
  [ -f "$archive" ] || {
    echo "Missing archive, skip: $archive"
    return 0
  }
  mkdir -p "$dest"
  tar -xf "$archive" -C "$dest"
}

archive_dest() {
  local archive="$1"
  case "$archive" in
    *.tar.gz)
      printf '%s\n' "${archive%.tar.gz}"
      ;;
    *.tgz)
      printf '%s\n' "${archive%.tgz}"
      ;;
    *.tar)
      printf '%s\n' "${archive%.tar}"
      ;;
    *.zip)
      printf '%s\n' "${archive%.zip}"
      ;;
    *)
      printf '%s\n' "$archive.extracted"
      ;;
  esac
}

extract_archive_once() {
  local archive="$1"
  local dest="$2"
  local marker="$dest/.post_extract_done"

  if [ -f "$marker" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "Already extracted, skip: $archive -> $dest"
    return 0
  fi

  mkdir -p "$dest"
  case "$archive" in
    *.zip)
      unzip -q -n "$archive" -d "$dest"
      ;;
    *.tar|*.tar.gz|*.tgz)
      tar -xf "$archive" -C "$dest"
      ;;
    *)
      echo "Unsupported archive, skip: $archive"
      return 0
      ;;
  esac
  touch "$marker"
}

post_ntire_rgb2raw() {
  local root="$RGB_RAW_ROOT/NTIRE2025-rgb2raw"
  [ -d "$root" ] || {
    echo "NTIRE2025-rgb2raw not found, skip"
    return 10
  }

  # Official marcosv/rgb2raw structure:
  # train_rgb2raw.zip contains RAW/RGB training pairs as .npy/.png and metadata .pkl.
  # val_rgbs_in.zip and test_rgbs.zip contain RGB inputs for the NTIRE challenge.
  local found=0
  if [ -f "$root/train_rgb2raw.zip" ]; then
    found=1
    extract_zip_to_dir "$root/train_rgb2raw.zip" "$root/train_rgb2raw"
  else
    echo "Missing: $root/train_rgb2raw.zip"
  fi
  if [ -f "$root/val_rgbs_in.zip" ]; then
    found=1
    extract_zip_to_dir "$root/val_rgbs_in.zip" "$root/val_rgbs_in"
  else
    echo "Missing: $root/val_rgbs_in.zip"
  fi
  if [ -f "$root/test_rgbs.zip" ]; then
    found=1
    extract_zip_to_dir "$root/test_rgbs.zip" "$root/test_rgbs"
  else
    echo "Missing: $root/test_rgbs.zip"
  fi
  [ "$found" -eq 1 ] || return 10

  {
    echo "Official source: https://huggingface.co/datasets/marcosv/rgb2raw"
    echo "Post-processed layout:"
    echo "  train_rgb2raw/  extracted training RAW/RGB pairs"
    echo "  val_rgbs_in/    extracted validation RGB inputs"
    echo "  test_rgbs/      extracted test RGB inputs"
    echo
    echo "File counts:"
    find "$root" -type f | sed "s#^$root/##" | awk '
      {
        n=split($0,a,".");
        ext=tolower(a[n]);
        count[ext]++;
      }
      END {
        for (ext in count) print ext, count[ext];
      }
    ' | sort
  } > "$root/POST_PROCESSING_SUMMARY.txt"
}

post_mit_adobe_fivek() {
  local root="$RGB_RAW_ROOT/MIT-Adobe-FiveK"
  local archive="$root/fivek_dataset.tar"
  local extracted="$root/extracted"
  [ -d "$root" ] || {
    echo "MIT-Adobe-FiveK not found, skip"
    return 10
  }
  [ -f "$archive" ] || {
    echo "Missing archive: $archive"
    return 10
  }

  if [ -f "$root/fivek_dataset.tar.sha1" ] && [ -f "$archive" ] && command -v sha1sum >/dev/null 2>&1; then
    local expected
    local actual
    expected="$(awk '{print $NF}' "$root/fivek_dataset.tar.sha1")"
    actual="$(sha1sum "$archive" | awk '{print $1}')"
    if [ "$expected" != "$actual" ]; then
      echo "FiveK SHA1 mismatch: expected $expected got $actual" >&2
      exit 2
    fi
  fi

  # Official FiveK archive includes DNG inputs, Lightroom catalog/renditions,
  # semantic metadata, and license files. Keep the official archive and extract
  # under extracted/ without flattening names.
  extract_tar_to_dir "$archive" "$extracted"

  {
    echo "Official source: https://data.csail.mit.edu/graphics/fivek/"
    echo "Official archive: fivek_dataset.tar"
    echo "Post-processed layout:"
    echo "  extracted/      official archive contents, not renamed or flattened"
    echo
    echo "Top-level extracted entries:"
    find "$extracted" -maxdepth 2 -mindepth 1 -printf "%y %p\n" | sed "s# $root/# #" | head -200
  } > "$root/POST_PROCESSING_SUMMARY.txt"
}

post_top_level_archives() {
  # Known archives from earlier benchmark downloads. These are extracted into
  # same-name folders while preserving archive files for provenance/re-run.
  extract_zip_to_dir "$DATASET_ROOT/Chameleon.zip" "$DATASET_ROOT/Chameleon"
  extract_zip_to_dir "$DATASET_ROOT/WildRF.zip" "$DATASET_ROOT/WildRF"
}

post_in_the_wild_archives() {
  [ -d "$IN_THE_WILD_ROOT" ] || {
    echo "in-the-wild not found, skip"
    return 10
  }

  # Current in-the-wild layout stores the archives under:
  # - Chameleon.zip and WildRF.zip at in-the-wild root
  # - RRDataset/*.tar.gz
  # - AIGIBench/{train,val,test}/*.zip
  # HF/Xet caches are skipped intentionally.
  local found=0
  while IFS= read -r archive; do
    found=1
    local dest
    dest="$(archive_dest "$archive")"
    echo "Extract: $archive -> $dest"
    extract_archive_once "$archive" "$dest"
  done < <(
    find "$IN_THE_WILD_ROOT" \
      -path "*/.cache/*" -prune -o \
      -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) \
      -print | sort
  )

  [ "$found" -eq 1 ] || {
    echo "No in-the-wild archives found"
    return 10
  }

  {
    echo "in-the-wild archive extraction summary"
    echo "Root: $IN_THE_WILD_ROOT"
    echo
    echo "Archives:"
    find "$IN_THE_WILD_ROOT" \
      -path "*/.cache/*" -prune -o \
      -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) \
      -printf "%p\t%s bytes\n" | sort
    echo
    echo "Top-level sizes:"
    du -sh "$IN_THE_WILD_ROOT"/* 2>/dev/null || true
  } > "$IN_THE_WILD_ROOT/POST_PROCESSING_SUMMARY.txt"
}

post_generic_archives() {
  # Conservative pass: only extract archives directly under dataset root or one
  # level below rgb-raw. Avoid recursively expanding nested archives from HF/Xet
  # caches or extracted dataset internals.
  while IFS= read -r archive; do
    case "$archive" in
      "$DATASET_ROOT/Chameleon.zip"|"$DATASET_ROOT/WildRF.zip"|"$RGB_RAW_ROOT/NTIRE2025-rgb2raw/"*|"$RGB_RAW_ROOT/MIT-Adobe-FiveK/fivek_dataset.tar")
        continue
        ;;
    esac

    local base
    local dest
    base="$(basename "$archive")"
    dest="${archive%.*}"
    case "$base" in
      *.tar.gz|*.tgz)
        dest="${archive%.tar.gz}"
        dest="${dest%.tgz}"
        extract_tar_to_dir "$archive" "$dest"
        ;;
      *.tar)
        extract_tar_to_dir "$archive" "$dest"
        ;;
      *.zip)
        extract_zip_to_dir "$archive" "$dest"
        ;;
    esac
  done < <(
    {
      find "$DATASET_ROOT" -maxdepth 1 -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \)
      [ -d "$RGB_RAW_ROOT" ] && find "$RGB_RAW_ROOT" -mindepth 2 -maxdepth 2 -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \)
    } | sort
  )
}

write_dataset_manifest() {
  local out="$DATASET_ROOT/DATASET_MANIFEST.txt"
  {
    echo "Generated: $(date '+%F %T')"
    echo "Root: $DATASET_ROOT"
    echo
    echo "Top-level sizes:"
    du -sh "$DATASET_ROOT"/* 2>/dev/null || true
    echo
    echo "Archive files:"
    find "$DATASET_ROOT" -type f \( -name "*.zip" -o -name "*.tar" -o -name "*.tar.gz" -o -name "*.tgz" \) -printf "%p\t%s bytes\n" | sort
    echo
    echo "Raw/RGB extension counts:"
    if [ -d "$RGB_RAW_ROOT" ]; then
      find "$RGB_RAW_ROOT" -type f | awk '
        {
          n=split($0,a,".");
          ext=tolower(a[n]);
          count[ext]++;
        }
        END {
          for (ext in count) print ext, count[ext];
        }
      ' | sort
    fi
    echo
    echo "In-the-wild extension counts:"
    if [ -d "$IN_THE_WILD_ROOT" ]; then
      find "$IN_THE_WILD_ROOT" -path "*/.cache/*" -prune -o -type f -print | awk '
        {
          n=split($0,a,".");
          ext=tolower(a[n]);
          count[ext]++;
        }
        END {
          for (ext in count) print ext, count[ext];
        }
      ' | sort
    fi
  } > "$out"
}

need_cmd find
need_cmd awk
need_cmd sed
need_cmd sort
need_cmd du
need_cmd unzip
need_cmd tar

run_step "01_top_level_known_archives" post_top_level_archives
run_step "02_in_the_wild_archives" post_in_the_wild_archives
run_step "03_ntire2025_rgb2raw" post_ntire_rgb2raw
run_step "04_mit_adobe_fivek" post_mit_adobe_fivek
run_step "05_generic_archives" post_generic_archives
run_step "06_manifest" write_dataset_manifest

log "Post-processing finished. Manifest: $DATASET_ROOT/DATASET_MANIFEST.txt"
