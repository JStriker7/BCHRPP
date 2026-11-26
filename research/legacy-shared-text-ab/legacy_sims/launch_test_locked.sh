#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo 'Usage: bash launch_test_locked.sh PHYSICAL_GPU FRESH_OUTPUT_DIR' >&2
  exit 2
fi
gpu="$1"; output="$2"
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
exec python -u "$root/evaluate_test_locked.py" \
  --source /root/autodl-tmp/sims_bchrpp_20261102/data/unaligned_39.pkl \
  --train-audit /root/sims_raw_reencode_20261103_a/audit.json \
  --test-audit /root/sims_raw_reencode_20261103_a/test_audit.json \
  --text-train /root/sims_raw_reencode_20261103_a_text \
  --audio-train /root/sims_raw_reencode_20261103_a_audio \
  --vision-train /root/sims_raw_reencode_20261103_a_vision \
  --text-test /root/sims_raw_reencode_20261103_a_text_test \
  --audio-test /root/sims_raw_reencode_20261103_a_audio_test \
  --vision-test /root/sims_raw_reencode_20261103_a_vision_test \
  --runs /root/autodl-tmp/results --output-dir "$output"
