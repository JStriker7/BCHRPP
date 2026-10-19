# Resume the paired whole-frame / face-crop experiment on wjy

The previous training script stored NumPy metric scalars from the teacher's
`metricsTop.py` in the best checkpoint. The training loop could finish, but
PyTorch's `weights_only=True` read at the end then failed. The patch preserves
the teacher's metric values, storing them as Python floats in new checkpoints.
It does not change the model, losses, feature files, or metric definitions.

The recovery command below uses `weights_only=False` **only when explicitly
requested**. This can execute code contained in a malicious checkpoint: use it
solely on checkpoints created by your own interrupted wjy training run. It
checks the expected feature path, seed, metric protocol and validation MAE,
leaves the original `.pt` untouched, then writes a separate `_recovered.pt`
and the missing per-seed `.json`. It infers `last_epoch` from the configured
60 epochs and 15-epoch patience; the selected weights and metrics are real.

## 1. Inspect before changing anything

```bash
cd /home/wjy/SYX/BCHRPP_RCSE_Lite_20261008
RUN=/mnt/data2t/wjy/bchrpp_mosi_20261014
for group in paired_wholeframe_gpu0 paired_facecrop_gpu0; do
  echo "=== $group ==="
  find "$RUN/results/$group" -maxdepth 1 -type f \
    \( -name 'mosi_lite_seed*.pt' -o -name 'mosi_lite_seed*.json' \
       -o -name 'results.csv' -o -name 'summary.csv' \) -printf '%f %s bytes\n' | sort
done
```

Do not assume both groups stopped at the same seed. A directory with
`summary.csv` is complete; do not resume it. A seed with `.json` is complete;
do not recover it. Never retrain a seed with an existing `.pt` and no `.json`.

## 2. Deploy the small patch and run tests

Upload `mosi_checkpoint_recovery_wjy_20261019.zip` to this project directory.
It contains only the training script, ablation runner, their two tests and
this document; it does not contain any model weights or feature file.

```bash
BACKUP="$RUN/results/script_backup_$(date +%Y%m%d_%H%M%S)"
mkdir "$BACKUP"
cp -p scripts/train_mosi_extract.py "$BACKUP/"
cp -p scripts/run_mosi_extract_ablations.py "$BACKUP/"
unzip -l mosi_checkpoint_recovery_wjy_20261019.zip
unzip -o mosi_checkpoint_recovery_wjy_20261019.zip
PY=/mnt/data2t/wjy/bchrpp_mosi_20261014/env/bin/python
CUDA_VISIBLE_DEVICES=0 "$PY" -m unittest discover -s tests -p 'test_metrics_top_mosi.py' -v
CUDA_VISIBLE_DEVICES=0 "$PY" -m unittest discover -s tests -p 'test_mosi_extract_resume.py' -v
```

If either test fails, stop and send the output. Do not install any new
dependencies for this patch, and do not set the face-crop-only `PYTHONPATH`.

## 3. Recover only interrupted seed-222 checkpoints

For each inspected group that has `mosi_lite_seed222.pt` but no
`mosi_lite_seed222.json`, run the corresponding command. If seed 128 or
2026 is also interrupted, replace `222` in both the file and `--seed`.
Do not run these commands on an unfamiliar checkpoint.

```bash
RUN=/mnt/data2t/wjy/bchrpp_mosi_20261014
OLD="$RUN/features/mosi_deberta_clip_clap.pt"
NEW="$RUN/features/mosi_deberta_clip_clap_facecrop_v1.pt"

# Only if paired_wholeframe_gpu0 seed 222 has .pt but no .json:
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_mosi_extract.py \
  --features "$OLD" \
  --checkpoint "$RUN/results/paired_wholeframe_gpu0/mosi_lite_seed222.pt" \
  --seed 222 --epochs 60 --patience 15 --recover-trusted-checkpoint

# Only if paired_facecrop_gpu0 seed 222 has .pt but no .json:
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/train_mosi_extract.py \
  --features "$NEW" \
  --checkpoint "$RUN/results/paired_facecrop_gpu0/mosi_lite_seed222.pt" \
  --seed 222 --epochs 60 --patience 15 --recover-trusted-checkpoint
```

An error during recovery must be investigated; never delete the checkpoint
or bypass the validation checks. A successful recovery shows the validation
and test metrics and creates a `_recovered.pt` plus `.json` in that group.

## 4. Continue each incomplete group

Run only after its interrupted seeds are recovered. The runner's `--resume`
checks each completed seed's JSON and checkpoint, skips it, then trains only
missing seeds with the same paired settings. It refuses existing checkpoints
without JSON, and refuses directories already containing `summary.csv`.

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_mosi_extract_ablations.py \
  --features "$OLD" --output-dir "$RUN/results/paired_wholeframe_gpu0" \
  --seeds 222,128,2026 --modes lite --epochs 60 --batch-size 24 \
  --hidden-dim 128 --queries 4 --topk 4 --lr 0.0003 --amp --resume

CUDA_VISIBLE_DEVICES=0 "$PY" scripts/run_mosi_extract_ablations.py \
  --features "$NEW" --output-dir "$RUN/results/paired_facecrop_gpu0" \
  --seeds 222,128,2026 --modes lite --epochs 60 --batch-size 24 \
  --hidden-dim 128 --queries 4 --topk 4 --lr 0.0003 --amp --resume
```

Skip either command if that group's `summary.csv` already exists. Send the
two `summary.csv` files (or their contents) for same-seed teacher-metric
comparison. Do not compare a newly trained face-crop run against only the
historical read-only checkpoint rescore and call that a preprocessing gain.
