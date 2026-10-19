# MOSI evaluation protocol for the offline extract pipeline

`scripts/train_mosi_extract.py` now calls the byte-identical copy of
`English/utils/metricsTop.py` in `metrics/metricsTop.py` using
`MetricsTop("regression").getMetics("MOSI")`. Do not use the classification
variant for regression checkpoints. The original `metrics/evaluation.py`
remains untouched for the older pipelines.

The teacher's keys are `MAE`, `Corr`, `Mult_acc_7`, `Mult_acc_5`,
`Has0_acc_2`, `Has0_F1_score`, `Non0_acc_2`, and `Non0_F1_score`.
The previous `acc2`/`f1` correspond to the Has0 pair. `Mult_acc_7`
uses the same clipping and NumPy rounding rule as the previous `acc7`,
but all teacher outputs are rounded to four decimals. MAE and Corr may
also differ slightly due to float32 computation and rounding. The Non0
pair excludes samples whose ground-truth label is exactly zero. MSE is
not part of this teacher report. If predictions are constant, the teacher's
correlation can be NaN whereas the previous helper returned zero; this is
reported as-is to keep the teacher's implementation exact.

Checkpoint selection still uses unrounded validation MAE at the previous
float64 precision, not the rounded `MAE` in the report. `--evaluate`
recomputes both validation and test with the teacher metric and does not
write a checkpoint or JSON file. It also shows the validation metrics
previously stored in the checkpoint. Do not overwrite the original
`results/ablations` directory: the new ablation runner has a different
CSV schema and requires a fresh output directory.

The wjy baseline audit on October 19, 2026 found 1284/229/686
train/valid/test samples. All 2199 samples used the extractor's uniform
alignment fallback; the saved audio and visual source paths existed on
wjy. The three-seed `lite` mean was 0.3357628766 test acc7 and
0.9534424758 test MAE under the previous metric implementation. These
are historical results, not a newly trained teacher-metric baseline.

## Read-only rescore of existing lite checkpoints

On October 19, 2026, wjy ran `--evaluate` with `CUDA_VISIBLE_DEVICES=-1`
after the first GPU attempt failed during CUDA initialization (device 2).
The saved checkpoints were not retrained or overwritten. The new metrics
were recomputed from validation and test predictions on CPU:

| Seed | Best epoch | Test Mult_acc_7 | Test MAE | Test Corr | Test Has0_acc_2 | Test Non0_acc_2 | Test Mult_acc_5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 222 | 28 | 0.3484 | 0.9189 | 0.6869 | 0.7813 | 0.7942 | 0.3965 |
| 128 | 33 | 0.3397 | 0.9449 | 0.6621 | 0.7697 | 0.7774 | 0.3863 |
| 2026 | 24 | 0.3192 | 0.9966 | 0.6455 | 0.7551 | 0.7607 | 0.3673 |
| Mean | - | 0.335767 | 0.953467 | 0.664833 | 0.768700 | 0.777433 | 0.383367 |

The original test acc7 mean was 0.335763, so the rounded teacher score
does not demonstrate a performance improvement. Non0 and Has0 Acc-2
measure different subsets and should not be compared as an uplift.
