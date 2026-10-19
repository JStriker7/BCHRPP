#!/usr/bin/env bash
# =============================================================================
# BCHR++ RCSE-Lite on CMU-MOSI — end-to-end run script (run on the Linux server)
#
#   Step 1  check the MOSI data layout (no model loaded)
#   Step 2  offline feature extraction via the UNMODIFIED extract code
#   Step 3  validate the extracted .pt (feature contract)
#   Step 4  train RCSE-Lite + evaluate test once (acc7 / mae / corr / acc2 / f1)
#
# Usage:
#   1. Edit the paths under "1. paths you MUST edit" below.
#   2. bash run_mosi_extract.sh
#
# The extractor is run from the `extract/` directory; training runs from this
# project directory (BCHRPP_RCSE_Lite_20261008/).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# 1. paths you MUST edit for your server
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # this project dir
EXTRACT_DIR="$PROJECT_ROOT/../extract"                        # the extract/ folder

DATA_ROOT="/path/to/MOSI"                 # MOSI root: contains label.csv + wav/ + Raw/ (or processed/)
RUN_ROOT="/path/to/experiment_output"     # your writable output dir

# Pretrained models — set each to the real local path:
DEBERTA_MODEL="/path/to/deberta-v3-large"
CLIP_MODEL="/path/to/clip-vit-large-patch14"
CLAP_MODEL="/path/to/630k-audioset-best.pt"        # .pt -> requires laion-clap
CLAP_BERT_MODEL="/path/to/bert-base-uncased"
CLAP_ROBERTA_MODEL="/path/to/roberta-base"
CLAP_ROBERTA_TOKENIZER="/path/to/roberta-large"
CLAP_BART_MODEL="/path/to/bart-base"

# ---------------------------------------------------------------------------
# 2. derived paths (normally no need to edit)
# ---------------------------------------------------------------------------
FEATURES="$RUN_ROOT/features/mosi_deberta_clip_clap.pt"
CACHE_DIR="$RUN_ROOT/cache/mosi_offline"
CHECKPOINT="$RUN_ROOT/runs/mosi_extract_rcse_lite_best.pt"

mkdir -p "$RUN_ROOT/features" "$RUN_ROOT/cache" "$RUN_ROOT/runs"

# ---------------------------------------------------------------------------
# Step 1 — check layout (read-only, no model loaded)
# ---------------------------------------------------------------------------
echo "== [1/4] check MOSI data layout =="
python "$EXTRACT_DIR/extract_mosi_features.py" --data-dir "$DATA_ROOT" --check-layout

# ---------------------------------------------------------------------------
# Step 2 — offline feature extraction (UNCHANGED extract code)
# ---------------------------------------------------------------------------
echo "== [2/4] extract offline features =="
python "$EXTRACT_DIR/extract_mosi_features.py" \
  --data-dir "$DATA_ROOT" \
  --output "$FEATURES" \
  --cache-dir "$CACHE_DIR" \
  --deberta-model "$DEBERTA_MODEL" \
  --clip-model "$CLIP_MODEL" \
  --clap-model "$CLAP_MODEL" \
  --clap-bert-model "$CLAP_BERT_MODEL" \
  --clap-roberta-model "$CLAP_ROBERTA_MODEL" \
  --clap-roberta-tokenizer "$CLAP_ROBERTA_TOKENIZER" \
  --clap-bart-model "$CLAP_BART_MODEL"

# ---------------------------------------------------------------------------
# Step 3 — validate features
# ---------------------------------------------------------------------------
echo "== [3/4] check extracted features =="
python "$PROJECT_ROOT/scripts/train_mosi_extract.py" --features "$FEATURES" --check-features

# ---------------------------------------------------------------------------
# Step 4 — train + evaluate
# ---------------------------------------------------------------------------
echo "== [4/4] train + evaluate RCSE-Lite =="
python "$PROJECT_ROOT/scripts/train_mosi_extract.py" \
  --features "$FEATURES" \
  --checkpoint "$CHECKPOINT" \
  --epochs 60 --batch-size 24 --hidden-dim 128 \
  --queries 4 --topk 4 --lr 0.0003 --seed 222 --amp

# Re-evaluate a saved checkpoint on test only (optional):
# python "$PROJECT_ROOT/scripts/train_mosi_extract.py" \
#   --features "$FEATURES" --checkpoint "$CHECKPOINT" --evaluate
