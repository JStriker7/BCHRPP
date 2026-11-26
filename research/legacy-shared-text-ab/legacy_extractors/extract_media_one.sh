#!/usr/bin/env bash
set -euo pipefail

if (( $# != 4 )); then
  echo 'Usage: bash extract_media_one.sh PHYSICAL_GPU audio|vision smoke|screen|test FRESH_OUTPUT_DIR' >&2
  exit 2
fi
gpu="$1"; encoder="$2"; mode="$3"; output="$4"
[[ "$gpu" =~ ^[0-9]+$ ]] || { echo 'GPU must be a physical index' >&2; exit 2; }
export SIMS_SELECTED_GPU="$gpu"
python - <<'PY'
import os, re, subprocess
output = subprocess.check_output(['mthreads-gmi'], text=True)
gpu = int(os.environ['SIMS_SELECTED_GPU'])
rows = {int(i): (int(used), int(total)) for i, used, total in
        re.findall(r'(?m)^\s*(\d+)\s+MTT S4000\b[^\n]*?(\d+)MiB\((\d+)MiB\)', output)}
if gpu not in rows or rows[gpu][1] - rows[gpu][0] < 24 * 1024:
    raise SystemExit(f'Physical GPU {gpu} missing or less than 24 GiB free: {rows}')
print(f'Physical GPU {gpu}: {rows[gpu][1]-rows[gpu][0]} MiB free', flush=True)
PY
export MUSA_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu"
root="$(cd "$(dirname "$0")" && pwd)"
features="${SIMS_FEATURES:-/root/autodl-tmp/sims_bchrpp_20261102/data/unaligned_39.pkl}"
raw="${SIMS_RAW_DIR:-/root/autodl-tmp/sims_raw_media_b1c55}"
if [[ "$encoder" == audio ]]; then
  model="${SIMS_HUBERT_MODEL:-/root/autodl-tmp/models/chinese_hubert_large_90cb66}"
else
  model="${SIMS_CLIP_MODEL:-/root/autodl-tmp/models/modelscope_clip_vitl14_f14d10}"
fi
exec python -u "$root/extract_media.py" --features "$features" --raw-dir "$raw" \
  --model "$model" --encoder "$encoder" --mode "$mode" --output-dir "$output"
