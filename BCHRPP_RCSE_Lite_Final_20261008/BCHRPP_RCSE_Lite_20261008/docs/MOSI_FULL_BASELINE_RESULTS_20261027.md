# MOSI full-modality reference result

Recorded from the operator-provided three-seed comparison row on 2026-10-27.
Original per-seed CSV and checkpoints remain on the training server.
The aggregate values below have not been independently recomputed from the
per-seed artifacts in this repository. They must not be used to choose new
configurations or checkpoints.

- Dataset/protocol: MOSI, `metricsTop.py:regression/MOSI`.
- Feature file: `/mnt/data2t/wjy/bchrpp_mosi_20261014/features/mosi_deberta_clip_clap.pt`.
- Variant/seeds: `modality/full`, `222`, `128`, `2026`.
- Model: BCHR++ RCSE-Lite with fully unfrozen DeBERTa-v3-large text,
  CLAP audio, and CLIP whole-frame visual features.
- Text learning rate: `2e-5`; fusion learning rate: `3e-4`.
- Checkpoint selection: validation MAE for each seed; test not used.
- Server comparison file: `/mnt/data2t/wjy/bchrpp_mosi_20261014/results/mosi_modality_20261020_gpu0/comparison.csv`.
- Reproduction entry point: `scripts/run_mosi_modality_next.sh` (also runs
  three additional modality ablations).

| Metric | Validation mean | Validation sample SD | Test mean | Test sample SD |
| --- | ---: | ---: | ---: | ---: |
| MAE | 0.574600 | 0.018361 | 0.591933 | 0.009799 |
| Corr | 0.868200 | 0.005828 | 0.861267 | 0.002219 |
| Mult_acc_7 | 0.473033 | 0.032793 | 0.499000 | 0.012745 |
| Mult_acc_5 | 0.567700 | 0.028654 | 0.577233 | 0.012698 |
| Has0_acc_2 | 0.873400 | 0.000000 | 0.864933 | 0.004708 |
| Has0_F1_score | 0.873733 | 0.000252 | 0.864367 | 0.004727 |
| Non0_acc_2 | 0.901233 | 0.002714 | 0.884633 | 0.003828 |
| Non0_F1_score | 0.900900 | 0.002862 | 0.884567 | 0.003953 |

The server deployment and original per-seed CSV have not been compared
byte-for-byte with this source tree. Treat this as an attributed result record,
not a cryptographically verified reproduction.
