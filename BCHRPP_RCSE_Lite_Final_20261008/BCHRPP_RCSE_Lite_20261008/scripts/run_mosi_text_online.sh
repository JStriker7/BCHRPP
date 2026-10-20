#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN="${MOSI_RUN_DIR:?Set MOSI_RUN_DIR to the experiment output directory}"
PY="${MOSI_PYTHON:-$RUN/env/bin/python}"
FEATURES="${MOSI_FEATURES:-$RUN/features/mosi_deberta_clip_clap.pt}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
"$PY" -m unittest discover -s tests -p 'test_mosi_text_online.py' -v
"$PY" -u scripts/check_mosi_text_parity.py --features "$FEATURES" \
  --samples-per-split 3 --device cuda:0
for text_mode in frozen unfrozen; do
  RESULTS="$RUN/results/paired_text_online_${text_mode}_gpu0"
  if test -e "$RESULTS/summary.csv"; then
    echo "已完成：$RESULTS/summary.csv；不覆盖"
    continue
  fi
  extra=()
  if test -e "$RESULTS/results.csv"; then extra+=(--resume); fi
  "$PY" -u scripts/run_mosi_extract_ablations.py \
    --features "$FEATURES" --output-dir "$RESULTS" --modes lite \
    --text-mode "$text_mode" --seeds 222,128,2026 --epochs 60 \
    --batch-size 24 --hidden-dim 128 --queries 4 --topk 4 \
    --lr 0.0003 --amp "${extra[@]}"
done
