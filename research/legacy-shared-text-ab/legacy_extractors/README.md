# CH-SIMS raw audio/video encoder candidates

Official `unaligned_39.pkl` supplies labels, IDs and train/valid/test split.
The separately pinned `tamb2203579/CH-SIMS` Raw.zip contains all clips for
these IDs; `unpack_raw.py` verifies both SHA-256 hashes, split membership,
archive paths and disk headroom before unpacking into a fresh directory.

`extract_media_one.sh` selects a physical GPU at launch and verifies free
memory. `audio` uses `TencentGameMate/chinese-hubert-large` revision
`90cb660492214f687e60f5ca509b20edae6e75bd` with 16 kHz mono audio,
sampling up to 128 token states across the full clip. `vision` uses the
user-provided `AI-ModelScope/clip-vit-large-patch14` revision
`f14d10b70c22170418f57120d301a7017cf60012`, eight uniformly sampled
video frames and CLIP vision-pooled states. Encoders are frozen, outputs are
float16 feature caches with a true-length mask. Screen mode only builds
train/valid caches; test mode must wait until the model is locked on validation.
The precomputed 33/709-dimensional official audio/vision features remain
available separately for ablations. Do not compare protocols without noting
the changed raw-media encoder inputs.

All results are written to fresh directories; no source results are overwritten
or deleted. Model and archive licenses should be reviewed before any external
redistribution.

## Independent re-encoding from raw inputs

`extract_text_v2.sh` encodes the official `raw_text` transcripts (not the
precomputed 39x768 embeddings) with frozen Chinese RoBERTa-wwm-ext-large
weights. Its weight directory contains **no working vocabulary**: loading
`AutoTokenizer` there silently creates a five-token tokenizer and maps Chinese
characters to `[UNK]`. The new script instead uses the pinned 21,128-token
`bert-base-chinese/vocab.txt`, validates vocabulary and special-token IDs,
rejects any text longer than 64 tokens, and saves token IDs and masks along
with 64x1024 float16 hidden states. Existing official token IDs match this
vocabulary for most samples; differences include official 39-token truncation
and mixed Latin text. Audio and video are independently decoded from the
previously verified Raw.zip extraction; no old feature cache is read.
The first remote text script copy is retained for provenance but must **not**
be launched: a CPU smoke test found that constructing `BertTokenizer` with a
direct `vocab_file` argument also falls back to five tokens on this installed
Transformers version. The corrected v2 uses `BertTokenizer.from_pretrained`
on the verified local tokenizer directory and refuses to run unless the full
vocabulary and special-token IDs load correctly.

These are frozen feature caches only, not model-training runs. The 64-token
text, 128-step HuBERT, and 8-frame CLIP temporal grids are deliberately
different; use per-modality masks and align rows by the official clip IDs.
Fit any clipping/scaling using train only, then reuse its parameters for
valid and eventually test. Keep source-video overlap across the official
splits in mind: clip-disjoint does not mean group-disjoint.
