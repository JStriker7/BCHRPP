# BCHR++ I3 research snapshot

R19 is the locked I3 reference evaluated on MOSI and CH-SIMS with seeds 222, 128, and 2026. R26 is a preregistered CH-SIMS warm-start experiment. This snapshot contains its launcher and rule. R26 results are separate from the R19 results.

## Code lineage

The direct model ancestor is BCHRPP RCSE-Lite. The normalized SHA-256 of its fusion core is identical to the historical 49.90% core preserved at [43eb9c9](https://github.com/JStriker7/BCHRPP/tree/43eb9c9a3033270354fe9db7133ed206b3f32de1). The original Lite and historical base-loss implementations also have identical Python ASTs. The older BCHR++ Full v2.0 implementation has a different core with causal disentanglement, counterfactual contribution, cognitive competition, causal hypergraph transformation, and dual arbitration.

The first four-dataset experiments used the Lite core with a different input and training wrapper. The later legacy499 transfer used the historical full-modality policy built on the Lite core. The I3 series then changed the feature representation and training policy on MOSI and CH-SIMS. I3 uses independent text, audio, and visual sequences, concatenating local/global audio context and center/full-field visual views. R19 adds a sign auxiliary that excludes exact-zero targets, uses a text learning rate of 1e-5, and reduces vision dropout to 5% while text and audio stay at 10%. R20–R25 are CH-SIMS validation experiments using the R19 family. R26 starts from the corresponding R19 checkpoint for each seed and uses an eight-epoch, reduced-LR fine-tune with the untouched checkpoint in the selection pool.

## Contents and runtime

- The exact R19 trainer and core live under `model_stage_20261106/R19_i3_signnonzero/`. Their byte hashes match the test lock for both datasets.
- `run_i3_r19_signnonzero_20261108.py` builds the R19 trainer from the packaged I3 source. The locked evaluator requires the original validated checkpoint hashes and plan.
- `run_i3_r26_warmselect_20261109.py` verifies the R19 source and initialization checkpoint hashes before using them. The R26 rule specifies validation-only selection.
- See [R19 test results](R19_I3_STAGE_REVIEW_20261108.md), [historical four-dataset transfer](LEGACY499_FOUR_DATASET_RESULTS.md), and the [I3 input report](INPUT_BASELINE_I3_REPORT_20261106.md).

Use a separate work root for each dataset. The environment versions are in `ENVIRONMENT_LOCK.json`. The runtime expects the independently obtained pretrained models under `legacy499/models/` and the matching I3 feature cache under `input_alignment_followup_20261106/grouped_features_v1/`. Its manifest must match the dataset's feature lock. Raw media, benchmark metadata, feature caches, and model weights are external runtime inputs. For raw extraction, supply the matching source archive and metadata to `extract_input_baseline_i3.py`. Use its `--help` for inputs and the smoke option. Check available physical GPU memory and pass the GPU index explicitly. The training process uses logical `musa:0`.

```sh
python run_i3_r19_signnonzero_20261108.py --root /path/to/work-root --dataset sims --physical-gpu YOUR_GPU
python run_i3_r26_warmselect_20261109.py --root /path/to/work-root --physical-gpu YOUR_GPU
```

R26 requires the exact R19 checkpoints referenced by its rule. A new R19 retraining run is not guaranteed to reproduce those bytes. R19 reported results cover two datasets. The four-dataset report covers the earlier legacy499 transfer.
