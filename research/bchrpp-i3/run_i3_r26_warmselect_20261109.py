"""Run the CH-SIMS validation-only R19 warm-start fine-tuning."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--physical-gpu", required=True)
args = parser.parse_args()
root = args.root
stage = root / "model_stage_20261106"
baseline = stage / "R19_i3_signnonzero"
output = stage / "R26_i3_warmselect"
seeds = (222, 128, 2026)
checkpoint_paths = [baseline / "results/TAV" / str(seed) / "best.pt" for seed in seeds]
required = sum(path.stat().st_size for path in checkpoint_paths) + 12 * 1024**3
assert shutil.disk_usage(root).free >= required, "Insufficient checkpoint space"

env = dict(os.environ, MUSA_VISIBLE_DEVICES=args.physical_gpu,
           CUDA_VISIBLE_DEVICES=args.physical_gpu, OMP_NUM_THREADS="4",
           MKL_NUM_THREADS="4", PYTHONDONTWRITEBYTECODE="1")
subprocess.run(["mthreads-gmi"], check=True)
subprocess.run(
    ["python", "-c", "import torch,torch_musa; f,t=torch.musa.mem_get_info(); "
     "assert torch.musa.device_count()==1 and f>12*1024**3; print('SELECTED_GPU_FREE',f)"],
    env=env, check=True,
)

def digest(content):
    return hashlib.sha256(content).hexdigest()

trainer = (baseline / "train_model.py").read_bytes()
core = (baseline / "vision_retain_core.py").read_bytes()
assert digest(trainer) == "f67105a8b5fce6aa3b238218e83f69ac1b96fd44292d35f2cfac8235d56c3b20"
assert digest(core) == "bf75df41ce28b378312668cbe278976347e028aaff069bbb9bc9710b0d22c016"
source = trainer.decode()
old_output = "model_stage_20261106/R19_i3_signnonzero/results"
assert source.count(old_output) == 1
source = source.replace(old_output, "model_stage_20261106/R26_i3_warmselect/results")
selection_rule = "validation joint F1/Acc3 rank with seed-matched MAE/Acc2 guards"
old = "best=float('inf');best_epoch=0;state=None;best_metrics=None"
new = """best=float('inf');best_epoch=0;state=None;best_metrics=None
lowest_mae=float('inf');lowest_mae_epoch=0;best_key=None
seed_ref=json.loads((Path(__file__).resolve().parents[2]/'R26_CH_SIMS_VALIDATION_RULE_20261109.json').read_text())['seed_reference'][str(a.seed)]
"""
assert source.count(old) == 1
source = source.replace(old, new)
old = " if mae<best:"
new = """ if mae<lowest_mae:
  lowest_mae=mae;lowest_mae_epoch=epoch
 f1=metrics['F1_score']-.7946
 acc3=metrics['Mult_acc_3']-.7083333333333334
 eligible=mae<=1.02*seed_ref[0] and metrics['Mult_acc_2']>=seed_ref[1]-.005-1e-12
 key=(1,min(f1,acc3),(f1+acc3)/2,-mae) if eligible else (0,0.,0.,-mae)
 if best_key is None or key>best_key:
  best_key=key"""
assert source.count(old) == 1
source = source.replace(old, new)
assert source.count("if epoch-best_epoch>=15:break") == 1
source = source.replace("if epoch-best_epoch>=15:break", "if epoch-lowest_mae_epoch>=15:break")
source = source.replace("'selection':'minimum validation raw MAE'", "'selection':" + repr(selection_rule) + " if best_key[0] else 'fallback minimum validation raw MAE'")
replacements = {"model=MOSIOnlineText(encoder,fusion,train_text=True).float().to('musa:0');opt=": "model=MOSIOnlineText(encoder,fusion,train_text=True).float()\nwarm_checkpoint=r.parent/'model_stage_20261106/R19_i3_signnonzero/results/TAV'/str(a.seed)/'best.pt'\nwarm_rule=json.loads((r.parent/'R26_CH_SIMS_VALIDATION_RULE_20261109.json').read_text())\nwarm_digest=hashlib.sha256()\nwith warm_checkpoint.open('rb') as file:\n for block in iter(lambda:file.read(8*1024**2),b''):\n  warm_digest.update(block)\nassert warm_digest.hexdigest()==warm_rule['initialization_checkpoint_sha256_by_seed'][str(a.seed)]\nwarm_state=torch.load(warm_checkpoint,map_location='cpu')\nmodel.load_state_dict(warm_state['state_dict'],strict=True)\ndel warm_state\nmodel=model.to('musa:0');opt=", "'lr':3e-4": "'lr':3e-5", "'lr':1e-5": "'lr':1e-6", "60*len(loaders['train'])": "8*len(loaders['train'])", "'epochs':60,'patience':15": "'epochs':8,'patience':None", "'fusion_lr':3e-4,'text_lr':1e-5": "'fusion_lr':3e-5,'text_lr':1e-6", 'for epoch in range(1,61):': 'for epoch in range(0,9):', 'loss=train_epoch();mae': 'loss=train_epoch() if epoch else None;mae', ' if epoch-lowest_mae_epoch>=15:break': '', "'max_epochs':60,'patience':15": "'max_epochs':8,'patience':None", "with (out/'manifest.json').open('x')": "record['initialization_checkpoint_sha256']=warm_digest.hexdigest()\nwith (out/'manifest.json').open('x')"}
for old,new in replacements.items():
    assert source.count(old)==1,old
    source=source.replace(old,new)
compile(source, str(output / "train_model.py"), "exec")
output.mkdir(exist_ok=False)
(output / "vision_retain_core.py").write_bytes(core)
(output / "train_model.py").write_bytes(source.encode())
rule = json.loads(Path(__file__).with_name("R26_CH_SIMS_VALIDATION_RULE_20261109.json").read_text())
with (output / "PLAN_LOCK.json").open("x") as file:
    json.dump(dict(rule, trainer_sha256=digest(source.encode()), core_sha256=digest(core)), file, indent=2)

command = ["python", "-u", str(output / "train_model.py"), "--root",
           str(root / "legacy499"), "--dataset", "sims", "--modality-control", "TAV"]
with (output / "smoke_batch24.log").open("xb") as file:
    subprocess.run(command + ["--seed", "222", "--smoke"], env=env, stdout=file,
                   stderr=subprocess.STDOUT, check=True)
print("SMOKE_PASSED", flush=True)

results = []
for seed in seeds:
    with (output / f"train_seed{seed}.log").open("xb") as file:
        subprocess.run(command + ["--seed", str(seed)], env=env, stdout=file,
                       stderr=subprocess.STDOUT, check=True)
    folder = output / "results/TAV" / str(seed)
    result = json.loads((folder / "VALIDATION_RESULT.json").read_text())
    checksum = hashlib.sha256()
    with (folder / "best.pt").open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024**2), b""):
            checksum.update(block)
    assert checksum.hexdigest() == result["checkpoint_sha256"]
    assert not result["test_scored"]
    rows = [json.loads(line) for line in (folder / "epochs.jsonl").read_text().splitlines()]
    epoch = next(row for row in rows if row["epoch"] == result["best_epoch"])
    assert epoch["validation_raw_mae"] == result["validation_raw_mae"]
    results.append(dict(result, validation=epoch["validation"]))
with (output / "VALIDATION_ROUND_COMPLETE.json").open("x") as file:
    json.dump({"seeds": results, "test_scored": False}, file, indent=2)
