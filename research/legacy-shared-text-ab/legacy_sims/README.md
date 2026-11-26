# CH-SIMS raw-input re-encoding and validation-only BCHR++

This experiment does **not** train on the official precomputed `text`, `audio`,
`vision`, or `text_bert` arrays. The pinned source pickle is SHA-256
`c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`.

## Preprocessing protocol

1. Preserve original clip IDs and official train/valid splits (1,368/456).
   Check count, uniqueness, ID order, source hash, and raw-media provenance.
   The 59 overlapping source-video groups are disclosed; clip IDs do not overlap.
2. Encode each original transcript in `raw_text` with the frozen Chinese
   RoBERTa-large checkpoint and verified 21,128-token Chinese vocabulary;
   pad to 64 tokens and save the token mask. Fail on any truncation or fallback
   to a five-token vocabulary.
3. Decode each original MP4, extract audio with frozen HuBERT (128 steps) and
   video with frozen CLIP (8 frames); save float16 features plus valid-step masks.
   These are **different** from CICA's official feature protocol.
4. Audit each train/valid cache against source IDs, shapes, finite values,
   padding, file SHA-256 and raw clip SHA-256. Compute channel mean/std only
   on valid **training** steps for audio/video; apply the same statistics to
   validation, then zero the padding. Keep text embeddings unstandardized.
5. Train BCHR++ with frozen re-encoded features and official training labels.
   Validation MAE alone selects checkpoints. Do not extract test features or
   use test labels for this candidate-selection stage.

The verified audit is `/root/sims_raw_reencode_20261103_a/audit.json`.
Train/valid cache roots are `/root/sims_raw_reencode_20261103_a_{text,audio,vision}`.
The GPU-checked smoke run passed two pretraining and two joint-training
steps. Three validation-only runs used the same configuration and wrote to
`/root/autodl-tmp/results/sims_raw_reencode_screen_{222,128,2026}_20261103_v2`.

| Seed | Best epoch | Valid MAE | Corr | Acc3 | Acc5 | Acc2 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 222 | 3 | 0.3345 | 0.7718 | 0.7039 | 0.4693 | 0.8180 | 0.8210 |
| 128 | 3 | 0.3291 | 0.7816 | 0.7237 | 0.4605 | 0.8202 | 0.8209 |
| 2026 | 6 | 0.3234 | 0.7785 | 0.7149 | 0.4846 | 0.8224 | 0.8263 |
| Mean | - | 0.3290 | 0.7773 | 0.7142 | 0.4715 | 0.8202 | 0.8227 |

These are **validation** results only, not a test comparison. In particular,
Acc3 and Acc2 remain below the target values quoted for CICA Table 2.

## Locked test evaluation

After the validation-selected epochs and checkpoint SHA-256 values were locked
in `LOCKED_TEST_PROTOCOL_20261103.md`, original test transcripts and MP4 clips
were encoded with the same frozen models. The separate test extraction audit
`/root/sims_raw_reencode_20261103_a/test_audit.json` verified 457 IDs, feature
shapes, masks, model hashes and clip hashes. No test statistics were fitted.
The final report is `/root/autodl-tmp/results/sims_raw_reencode_test_locked_20261103_v2/result.json`
(SHA-256 `e1d09f352fb1e4243b581d7e8f93b068c490959603f4b603d7f03b766f8d0959`).

| Seed | Test MAE | Corr | Acc3 | Acc5 | Acc2 | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 222 | 0.3213 | 0.8009 | 74.84% | 45.95% | 87.96% | 88.20% |
| 128 | 0.3346 | 0.7960 | 74.84% | 44.86% | 87.53% | 87.61% |
| 2026 | 0.3084 | 0.8080 | 76.59% | 50.11% | 86.21% | 86.51% |
| Mean | 0.3214 | 0.8016 | 75.42% | 46.97% | 87.23% | 87.44% |

Four of five CICA Table 2 numerical thresholds are met. Mean Acc3 is 0.95
percentage points below 76.37%; this is **not** a five-metric success.
Acc5 is included for completeness; CICA Table 2 does not report a CH-SIMS
Acc5 value for comparison.
Different HuBERT/CLIP feature extraction also prevents a controlled
same-protocol comparison. The first evaluation produced a partial JSON due
only to NumPy boolean serialization; it was preserved, and the same three
checkpoint hashes were re-evaluated without any model or metric change in
the fresh `_v2` result directory.

The three caches consume approximately 715 MiB on the system disk. The
50-GiB data volume had about 2.7 GiB free when training started; that is
enough for this small BCHR++ checkpoint, but not a comfortable reserve for
additional large model checkpoints or test media caches. Increase the data
volume to around 100 GiB total before a larger experimental sweep (80 GiB
total is a constrained minimum). No existing experiment is removed.
