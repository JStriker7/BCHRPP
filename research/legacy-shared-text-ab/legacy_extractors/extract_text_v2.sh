#!/usr/bin/env bash
set -euo pipefail
if (( $# != 2 )) || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo 'Usage: bash extract_text_one.sh PHYSICAL_GPU FRESH_OUTPUT_DIR' >&2
  exit 2
fi
export SIMS_SELECTED_GPU="$1"
python - <<'PY'
import os, re, subprocess
rows = {int(i): (int(used), int(total)) for i, used, total in re.findall(
    r'(?m)^\s*(\d+)\s+MTT S4000\b[^\n]*?(\d+)MiB\((\d+)MiB\)',
    subprocess.check_output(['mthreads-gmi'], text=True))}
gpu = int(os.environ['SIMS_SELECTED_GPU'])
if gpu not in rows or rows[gpu][1] - rows[gpu][0] < 24 * 1024:
    raise SystemExit(f'GPU {gpu} missing or less than 24 GiB free: {rows}')
print(f'Physical GPU {gpu}: {rows[gpu][1]-rows[gpu][0]} MiB free', flush=True)
PY
export MUSA_VISIBLE_DEVICES="$1" CUDA_VISIBLE_DEVICES="$1"
root="$(cd "$(dirname "$0")" && pwd)"
exec python -u "$root/extract_text_v2.py" \
  --source /root/autodl-tmp/sims_bchrpp_20261102/data/unaligned_39.pkl \
  --model /root/autodl-tmp/sims_bchrpp_20261102/models/chinese-roberta-wwm-ext-large \
  --vocab /root/autodl-tmp/sims_bchrpp_20261102/models/bert-base-chinese/vocab.txt \
  --output-dir "$2"
