#!/usr/bin/env bash
set -euo pipefail

DATE_TAG="$(date +%Y%m%d)"
LOG_DIR="/data0/xzp/migration_logs"
MAIN_LOG="${LOG_DIR}/migration_${DATE_TAG}.log"
LOCK_DIR="${LOG_DIR}/migration_${DATE_TAG}.lock"

SRC_AIGC="/data1/xzp/AIGC"
SRC_DATASET="/data1/xzp/dataset"
DST_AIGC="/data0/xzp/AIGC"
DST_DATASET="/data0/xzp/dataset"
BACKUP_AIGC="/data1/xzp/AIGC_migrated_backup_${DATE_TAG}"
BACKUP_DATASET="/data1/xzp/dataset_migrated_backup_${DATE_TAG}"

mkdir -p "${LOG_DIR}" /data0/xzp
if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "Migration lock exists: ${LOCK_DIR}" >&2
  exit 1
fi
trap 'rmdir "${LOCK_DIR}" 2>/dev/null || true' EXIT

exec > >(tee -a "${MAIN_LOG}") 2>&1

echo "== migration started $(date -Is) =="
echo "source AIGC=${SRC_AIGC}"
echo "source dataset=${SRC_DATASET}"
echo "dest AIGC=${DST_AIGC}"
echo "dest dataset=${DST_DATASET}"

echo "== pre-check disk =="
df -h /data0 /data1

echo "== pre-check source sizes =="
du -sh "${SRC_AIGC}" "${SRC_DATASET}"

echo "== pre-check active old-path processes =="
ACTIVE_PROCS="$(ps -eo pid,etime,cmd | grep -E "/data1/xzp/AIGC|/data1/xzp/dataset|raw_like_stats|AIGIBench|CycleISP" | grep -v grep || true)"
if [[ -n "${ACTIVE_PROCS}" ]]; then
  echo "${ACTIVE_PROCS}"
  echo "Active AIGC/dataset-related processes exist. Stop migration before rsync."
  exit 2
fi
echo "none"

echo "== ensure destination parent =="
mkdir -p /data0/xzp

echo "== rsync AIGC $(date -Is) =="
rsync -aH --info=progress2 "${SRC_AIGC}/" "${DST_AIGC}/" 2>&1 | tee "${LOG_DIR}/rsync_AIGC.log"

echo "== rsync dataset $(date -Is) =="
rsync -aH --info=progress2 "${SRC_DATASET}/" "${DST_DATASET}/" 2>&1 | tee "${LOG_DIR}/rsync_dataset.log"

echo "== verify sizes =="
du -sh "${SRC_AIGC}" "${DST_AIGC}"
du -sh "${SRC_DATASET}" "${DST_DATASET}"

echo "== verify file counts =="
SRC_AIGC_COUNT="$(find "${SRC_AIGC}" -type f | wc -l)"
DST_AIGC_COUNT="$(find "${DST_AIGC}" -type f | wc -l)"
SRC_DATASET_COUNT="$(find "${SRC_DATASET}" -type f | wc -l)"
DST_DATASET_COUNT="$(find "${DST_DATASET}" -type f | wc -l)"
echo "AIGC files old=${SRC_AIGC_COUNT} new=${DST_AIGC_COUNT}"
echo "dataset files old=${SRC_DATASET_COUNT} new=${DST_DATASET_COUNT}"
if [[ "${SRC_AIGC_COUNT}" -ne "${DST_AIGC_COUNT}" ]]; then
  echo "AIGC file-count mismatch; abort before path update/symlink."
  exit 3
fi
if [[ "${SRC_DATASET_COUNT}" -ne "${DST_DATASET_COUNT}" ]]; then
  echo "dataset file-count mismatch; abort before path update/symlink."
  exit 4
fi

echo "== verify key directories =="
test -d "${DST_AIGC}/raw_like_stats_v0"
test -d "${DST_AIGC}/raw_like_stats_v1"
test -d "${DST_AIGC}/CycleISP"
test -d "${DST_DATASET}/in-the-wild/AIGIBench"
echo "key directories ok"

echo "== verify AIGIBench image counts =="
SRC_AIGIBENCH_IMAGES="$(find "${SRC_DATASET}/in-the-wild/AIGIBench" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)"
DST_AIGIBENCH_IMAGES="$(find "${DST_DATASET}/in-the-wild/AIGIBench" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.webp" \) | wc -l)"
echo "AIGIBench images old=${SRC_AIGIBENCH_IMAGES} new=${DST_AIGIBENCH_IMAGES}"
if [[ "${SRC_AIGIBENCH_IMAGES}" -ne "${DST_AIGIBENCH_IMAGES}" ]]; then
  echo "AIGIBench image-count mismatch; abort before path update/symlink."
  exit 5
fi

echo "== update active path dependencies in new AIGC =="
python3 - <<'PY'
from pathlib import Path

root = Path("/data0/xzp/AIGC")
changed_log = Path("/data0/xzp/migration_logs/path_rewrites_active_files.txt")
remaining_log = Path("/data0/xzp/migration_logs/remaining_old_refs_active_files.txt")
skip_parts = {".git", "__pycache__", "outputs"}
active_suffixes = {".py", ".yaml", ".yml", ".sh", ".toml", ".json", ".cfg", ".ini", ".md"}
old_new = [
    (b"/data1/xzp/AIGC", b"/data0/xzp/AIGC"),
    (b"/data1/xzp/dataset", b"/data0/xzp/dataset"),
]

def is_active_file(path: Path) -> bool:
    rel_parts = set(path.relative_to(root).parts)
    if rel_parts & skip_parts:
        return False
    if path.suffix.lower() in active_suffixes:
        return True
    if path.suffix.lower() == ".csv" and "manifests" in rel_parts:
        return True
    return False

changed = []
remaining = []
for path in root.rglob("*"):
    if not path.is_file() or not is_active_file(path):
        continue
    try:
        data = path.read_bytes()
    except OSError:
        continue
    new_data = data
    for old, new in old_new:
        new_data = new_data.replace(old, new)
    if new_data != data:
        path.write_bytes(new_data)
        changed.append(str(path))
    if b"/data1/xzp/AIGC" in new_data or b"/data1/xzp/dataset" in new_data:
        remaining.append(str(path))

changed_log.write_text("\n".join(changed) + ("\n" if changed else ""), encoding="utf-8")
remaining_log.write_text("\n".join(remaining) + ("\n" if remaining else ""), encoding="utf-8")
print(f"changed active files: {len(changed)}")
print(f"remaining active old-path files: {len(remaining)}")
if remaining:
    print("WARNING: active files still contain old paths; see", remaining_log)
PY

echo "== scan remaining old-path references in new AIGC =="
grep -R -n -E "/data1/xzp/AIGC|/data1/xzp/dataset" "${DST_AIGC}" \
  --include="*.py" \
  --include="*.yaml" \
  --include="*.yml" \
  --include="*.json" \
  --include="*.sh" \
  --include="*.toml" \
  --include="*.csv" \
  > "${LOG_DIR}/remaining_old_refs_all_text.txt" || true
wc -l "${LOG_DIR}/remaining_old_refs_all_text.txt"

echo "== rename old folders as backups and create symlinks =="
if [[ -e "${BACKUP_AIGC}" || -L "${BACKUP_AIGC}" ]]; then
  BACKUP_AIGC="${BACKUP_AIGC}_$(date +%H%M%S)"
fi
if [[ -e "${BACKUP_DATASET}" || -L "${BACKUP_DATASET}" ]]; then
  BACKUP_DATASET="${BACKUP_DATASET}_$(date +%H%M%S)"
fi
mv "${SRC_AIGC}" "${BACKUP_AIGC}"
mv "${SRC_DATASET}" "${BACKUP_DATASET}"
ln -s "${DST_AIGC}" "${SRC_AIGC}"
ln -s "${DST_DATASET}" "${SRC_DATASET}"

echo "== verify symlinks =="
test -L "${SRC_AIGC}"
test -L "${SRC_DATASET}"
echo "AIGC symlink -> $(readlink -f "${SRC_AIGC}")"
echo "dataset symlink -> $(readlink -f "${SRC_DATASET}")"
test "$(readlink -f "${SRC_AIGC}")" = "${DST_AIGC}"
test "$(readlink -f "${SRC_DATASET}")" = "${DST_DATASET}"

echo "== smoke checks =="
ls "${DST_AIGC}" | sed -n '1,80p'
ls "${DST_DATASET}/in-the-wild/AIGIBench" | sed -n '1,80p'
ls "${SRC_AIGC}" | sed -n '1,40p'
ls "${SRC_DATASET}/in-the-wild/AIGIBench" | sed -n '1,40p'

echo "== /data0 references in configs =="
grep -R -n -E "/data0/xzp/AIGC|/data0/xzp/dataset" \
  "${DST_AIGC}/configs" "${DST_AIGC}"/raw_like_stats_v*/configs \
  > "${LOG_DIR}/data0_refs_in_configs.txt" || true
wc -l "${LOG_DIR}/data0_refs_in_configs.txt"

echo "== final disk =="
df -h /data0 /data1

cat > "${LOG_DIR}/migration_report_${DATE_TAG}.txt" <<REPORT
Migration report generated $(date -Is)

Old paths:
  ${SRC_AIGC}
  ${SRC_DATASET}
New paths:
  ${DST_AIGC}
  ${DST_DATASET}
Backups:
  ${BACKUP_AIGC}
  ${BACKUP_DATASET}

Copied file counts:
  AIGC old=${SRC_AIGC_COUNT} new=${DST_AIGC_COUNT}
  dataset old=${SRC_DATASET_COUNT} new=${DST_DATASET_COUNT}
  AIGIBench images old=${SRC_AIGIBENCH_IMAGES} new=${DST_AIGIBENCH_IMAGES}

Symlinks:
  ${SRC_AIGC} -> $(readlink -f "${SRC_AIGC}")
  ${SRC_DATASET} -> $(readlink -f "${SRC_DATASET}")

Changed active files:
  ${LOG_DIR}/path_rewrites_active_files.txt

Remaining old-path refs:
  active-only: ${LOG_DIR}/remaining_old_refs_active_files.txt
  all text/history scan: ${LOG_DIR}/remaining_old_refs_all_text.txt

Logs:
  ${MAIN_LOG}
  ${LOG_DIR}/rsync_AIGC.log
  ${LOG_DIR}/rsync_dataset.log
REPORT

echo "== migration completed $(date -Is) =="
echo "report: ${LOG_DIR}/migration_report_${DATE_TAG}.txt"
