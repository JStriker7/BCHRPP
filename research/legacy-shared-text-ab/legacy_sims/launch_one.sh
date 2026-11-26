#!/usr/bin/env bash
set -euo pipefail
if (( $# < 4 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo 'Usage: bash launch_one.sh PHYSICAL_GPU SEED smoke|screen FRESH_OUTPUT_DIR [flags]' >&2
  exit 2
fi
gpu="$1"; seed="$2"; mode="$3"; output="$4"; shift 4
export SIMS_SELECTED_GPU="$gpu"
python - <<'PY'
import os, re, subprocess
rows = {int(i): (int(used), int(total)) for i, used, total in re.findall(
    r'(?m)^\s*(\d+)\s+MTT S4000\b[^\n]*?(\d+)MiB\((\d+)MiB\)',
    subprocess.check_output(['mthreads-gmi'], text=True))}
gpu = int(os.environ['SIMS_SELECTED_GPU'])
if gpu not in rows or rows[gpu][1] - rows[gpu][0] < 16 * 1024:
    raise SystemExit(f'GPU {gpu} unavailable or <16GiB free: {rows}')
print(f'Physical GPU {gpu}: {rows[gpu][1]-rows[gpu][0]} MiB free', flush=True)
PY
export MUSA_VISIBLE_DEVICES="$gpu" CUDA_VISIBLE_DEVICES="$gpu"
root="$(cd "$(dirname "$0")" && pwd)"
exec python -u "$root/train_sims.py" \
  --source /root/autodl-tmp/sims_bchrpp_20261102/data/unaligned_39.pkl \
  --text-cache /root/sims_raw_reencode_20261103_a_text \
  --audio-cache /root/sims_raw_reencode_20261103_a_audio \
  --vision-cache /root/sims_raw_reencode_20261103_a_vision \
  --audit /root/sims_raw_reencode_20261103_a/audit.json \
  --output-dir "$output" --seed "$seed" --mode "$mode" "$@"
