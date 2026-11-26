# Locked CH-SIMS v1 test evaluation protocol

This protocol was fixed after the three validation-only runs and before
extracting test features. No training, checkpoint selection, threshold tuning,
model selection, or hyperparameter changes are permitted from test feedback.

- Model: frozen Chinese RoBERTa-large raw-text embeddings + HuBERT audio +
  CLIP video; BCHR++ fusion and CLS residual. Original test transcripts and
  MP4 clips will use exactly the train/valid extraction settings.
- Source pickle SHA-256: `c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f`.
- Train/valid audit SHA-256: `202ae819690d5f781ef6a9fa5288474b9f1b2d3e4abde20c9ea594c999c06ecb`.
- Frozen training source SHA-256: `3546e3066d4856691161e3da9d7a59191dfce5b307baba4d1cae34d7cffbe2b1`.
- Seeds/checkpoints: 222 (epoch 3), 128 (epoch 3), 2026 (epoch 6), each
  `best.pt` in `/root/autodl-tmp/results/sims_raw_reencode_screen_<seed>_20261103_v2`.
  All three checkpoints were selected exclusively by validation MAE.
  Locked checkpoint SHA-256 by seed: 222 `ca5f519e65e48bef12b02b3d38381c5ef132853dae259998842d782b647eeca4`,
  128 `03a29c7d70ad3644510ad21e85ffd7b60ae61deddb22d3f004bd93a93c114fd0`,
  2026 `fabb94295e79411c5c0b01a49cb09d50ee673dc42a9a701de0787463370a1413`.
- Evaluation: apply train-only HuBERT/CLIP channel statistics from the audit,
  leave text unstandardized, preserve masks and official test ID ordering.
  Report each seed's unmodified `metricsTop.py` SIMS regression metrics and
  their arithmetic mean. No ensemble, calibration, or test-dependent selection.
- Comparison: CICA Table 2 test thresholds MAE < 0.378, Corr > 0.754,
  Acc3 > 0.7637, Acc2 > 0.8600, F1 > 0.8568, all strict. Our re-encoded
  HuBERT/CLIP feature protocol differs from CICA's official features.
