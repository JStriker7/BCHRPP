#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

RUN="${MOSI_RUN_DIR:-/mnt/data2t/wjy/bchrpp_mosi_20261014}"
PY="${MOSI_PYTHON:-$RUN/env/bin/python}"
FEATURES="${MOSI_FEATURES:-$RUN/features/mosi_deberta_clip_clap.pt}"
MODEL="${MOSI_TEXT_MODEL:-/mnt/data26t/model/deberta-v3-large}"
OUT="${MOSI_OUTPUT_ROOT:-$RUN/results/mosi_modality_20261020_gpu0}"
LOG="${MOSI_LOG:-$RUN/results/mosi_modality_20261020_gpu0.log}"
export CUDA_VISIBLE_DEVICES=0

for entry in "$PY" "$FEATURES"; do
  if [[ ! -f "$entry" ]]; then echo "缺少必需文件：$entry" >&2; exit 1; fi
done
if [[ ! -d "$MODEL" ]]; then echo "缺少文本模型：$MODEL" >&2; exit 1; fi
mkdir -p "$(dirname "$LOG")"
run() { "$@" 2>&1 | tee -a "$LOG"; }

common=(--trial modality --features "$FEATURES"
        --expected-text-model "$MODEL" --output-root "$OUT")
run "$PY" -u scripts/mosi_followup.py "${common[@]}" --preflight-only
run "$PY" -m unittest discover -s tests -v
run "$PY" -u scripts/check_mosi_text_parity.py --features "$FEATURES" \
    --expected-text-model "$MODEL" --samples-per-split 3 --device cuda:0
run "$PY" -u scripts/mosi_followup.py "${common[@]}"
