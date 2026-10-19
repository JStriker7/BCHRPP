# MOSI trial 1: crop faces before the existing CLIP vision encoder

This is an isolated feature-preprocessing experiment. The trusted original
`mosi_deberta_clip_clap.pt` supplies the same 1284/229/686 splits, labels,
words, DeBERTa text, CLAP audio, and validity masks. The script changes only
the `visual` arrays and adds provenance metadata. It reuses the exact original
extractor's source reader, audio duration, 4 FPS sampling, uniform word windows,
CLIP processor and vision model. Face detection uses the teacher's MediaPipe
Tasks BlazeFace model, first detection with score >= 0.8, resized to 224x224.
Detection failure falls back to the original whole frame (not a black image).

The script hashes and checks the original extractor (`73101f9a...`), runs
uncropped CLIP parity checks against the original `.pt`, caches individual
samples for resumability, and refuses to overwrite the old or existing new
feature file. A dry run processes N samples *per split* without writing files.
Do not treat a successful extraction as a model-quality improvement: compare
three new training seeds on the same settings, selected by validation MAE, and
report the teacher `metricsTop.py` MOSI regression metrics.

## wjy environment preflight

```bash
cd /home/wjy/SYX/BCHRPP_RCSE_Lite_20261008
PY=/mnt/data2t/wjy/bchrpp_mosi_20261014/env/bin/python
ls -l ../extract/extract_mosi_features.py
sha256sum ../extract/extract_mosi_features.py
ls -ld /mnt/data26t/model/clip-vit-large-patch14
```

These checks passed on wjy. An unconstrained MediaPipe install would upgrade
the training environment's NumPy 1.26.4 to 2.2.6 and install OpenCV 5. Do not
install into the original venv. Use a separate package target with the same
NumPy 1.26.4, OpenCV 4.10.0, and Pillow 11.3.0 versions as the baseline. The
target directory must be new/empty. First inspect this dry-run:

```bash
TARGET=/mnt/data2t/wjy/bchrpp_mosi_20261014/facecrop_deps
"$PY" -m pip install --dry-run --target "$TARGET" \
  'mediapipe==0.10.32' 'numpy==1.26.4' \
  'opencv-contrib-python==4.10.0.84' 'pillow==11.3.0'
```

If the resolver accepts those pins, install **only** to the private target and
check which modules the extraction process will actually load:

```bash
"$PY" -m pip install --target "$TARGET" \
  'mediapipe==0.10.32' 'numpy==1.26.4' \
  'opencv-contrib-python==4.10.0.84' 'pillow==11.3.0'
PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" "$PY" -c \
  'import cv2, numpy, PIL, mediapipe; print("cv2", cv2.__version__, cv2.__file__); print("numpy", numpy.__version__); print("Pillow", PIL.__version__); print("mediapipe", mediapipe.__version__)'
"$PY" -c 'import cv2, numpy, PIL; print("baseline", cv2.__version__, numpy.__version__, PIL.__version__)'
```

Only set `PYTHONPATH` for the extraction command, not in the whole shell or
training command. Upload `mosi_face_crop_wjy_20261019_v2.zip` to the project
directory with VS Code, then unpack the new `scripts/`, `tests/`, and `docs/`
files there. If the first ZIP was already unpacked, this updated ZIP replaces
only those five experimental files. No old feature, checkpoint, or result
file is in the archive.

## GPU 0 dry run (no output files)

```bash
cd /home/wjy/SYX/BCHRPP_RCSE_Lite_20261008
PY=/mnt/data2t/wjy/bchrpp_mosi_20261014/env/bin/python
OLD=/mnt/data2t/wjy/bchrpp_mosi_20261014/features/mosi_deberta_clip_clap.pt
"$PY" -m unittest tests.test_face_crop_ops tests.test_reextract_mosi_faces -v
PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  "$PY" scripts/reextract_mosi_faces.py \
  --features "$OLD" --device cuda:0 --dry-run-samples 3
```

Each sample reports frames, accepted faces, fallback count, average cropped
face area, and the uncropped CLIP parity difference. The script aborts if
the uncropped features differ from the original by more than 0.001; do not
relax that tolerance to force the experiment through. Send the full dry-run
output for review before launching the full extraction. Small-sample coverage
alone is not the final coverage estimate.

## After the dry run passes

```bash
RUN=/mnt/data2t/wjy/bchrpp_mosi_20261014
NEW="$RUN/features/mosi_deberta_clip_clap_facecrop_v1.pt"
PYTHONPATH="$TARGET${PYTHONPATH:+:$PYTHONPATH}" CUDA_VISIBLE_DEVICES=0 \
  "$PY" scripts/reextract_mosi_faces.py \
  --features "$OLD" --output "$NEW" \
  --cache-dir "$RUN/cache/facecrop_v1" --device cuda:0
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_mosi_extract.py \
  --features "$NEW" --check-features
```

Review full train/valid/test face coverage and parity. If detection largely
fails, do not train a nearly unchanged feature file. For a controlled comparison,
train `lite` only with seeds 222,128,2026, the same hyperparameters and same
GPU visibility, using a *new* result directory. Do not replace old checkpoints
or the legacy `results/ablations` CSVs. Use only validation MAE to select a
checkpoint and avoid using test scores to tune the feature extractor.
