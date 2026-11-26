# Restored legacy SIMS method and full text freeze comparison

This snapshot preserves the earlier frozen RoBERTa / HuBERT / CLIP method, its MOSI transfer, and the current controlled text-encoder experiment. It does not replace R19.

- `legacy_sims/`: original fusion training, loss, CLS residual and historical evaluation scripts.
- `legacy_extractors/`: original pinned raw-media and text extraction scripts. Historical evaluation locks refer to historical checkpoints, not retrained files.
- `mosi_transfer/`: English frozen-encoder transfer and dataset adapters.
- `text_freeze_ab/`: current paired experiment. **Frozen** means no text weight updates; **unfrozen** means all text encoder layers are trainable, not just the last two layers.
- `results/`: completed restoration/transfer test results. The new freeze comparison is running; its results are not yet available.

## Controlled comparison

Each dataset uses seeds 222, 128, 2026. Both arms share the legacy fusion, zero-initialized CLS skip, train-only audio/video normalization, feature masks, seeded fusion initialization, five unimodal pretraining epochs, batch size 16, AdamW, fusion LR 3e-4 and weight decay 0.01. Joint training uses at most 20 epochs with patience 5, selecting minimum validation raw MAE. The text optimizer LR is 1e-5; frozen parameters receive no gradients or updates. Both arms keep encoder dropout disabled to isolate parameter updates. During pretraining, both arms use the identical frozen text. During joint training, the unfrozen arm recomputes contextual text with gradients. Frozen-cache/online parity and first/last encoder-block gradients are checked on real data before formal training.

SIMS has native T/A/V labels for auxiliary supervision. MOSI has no such labels, so it uses the overall training label for each auxiliary target. This adapter difference is recorded; within each dataset it is identical between frozen/unfrozen arms. Chinese and English encoders/tokenizers and official metric definitions are dataset adapters.

All six checkpoints for each dataset are locked before either arm is tested. Tests report per seed, mean and sample standard deviation. Historical test sets have already been viewed; tests do not choose the winning checkpoint.

## Runtime and launch

Tested environment: Python 3.10, PyTorch 2.11 with torch_musa / MUSA 5.2, transformers 5.13, NumPy and scikit-learn. The physical GPU must be explicitly chosen at launch; processes use internal `musa:0`. Data/models are excluded from Git.

The `--assets-root` directory must contain the verified official data and restored caches, using the layouts documented in the original extraction scripts and current runner. Existing deployments use `SIMS_LEGACY_RESTORE_20261125` and `LEGACY_SHARED_MOSI_20261126_v2`. Outputs must be fresh paths. At least three measured full text checkpoints plus compact frozen checkpoints and 12 GiB reserve are required.

```bash
python -u text_freeze_ab/train_compare.py --dataset sims --physical-gpu PHYSICAL_GPU --assets-root /path/to/BCHRPP_FRESH_20261104 --output /path/to/fresh_sims_output
python -u text_freeze_ab/train_compare.py --dataset mosi --physical-gpu PHYSICAL_GPU --assets-root /path/to/BCHRPP_FRESH_20261104 --output /path/to/fresh_mosi_output
```

Primary SIMS binary comparison uses zero-positive labels. The restored legacy historical scorer uses zero-negative labels; both conventions must be named explicitly when presenting historical results. MOSI reports Has0 and Non0 separately.

R19 remains available in the repository branch `codex/bchrpp-i3-research`, under `research/bchrpp-i3/model_stage_20261106/R19_i3_signnonzero/`.
