"""R1 shared full-budget model baseline on locked I3; train/validation only."""
import pathlib,argparse,json,hashlib,os,subprocess,statistics
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--dataset',choices=['mosi','sims'],required=True);p.add_argument('--physical-gpu',required=True);a=p.parse_args();r=a.root;import shutil;required=sum((r/'model_stage_20261106/R3_i3_textlr1e5/results/TAV'/str(seed)/'best.pt').stat().st_size for seed in [222,128,2026])+12*1024**3;assert shutil.disk_usage(r).free>required,'Insufficient disk for measured checkpoints plus12GiB reserve';print('DISK_BUDGET',required,shutil.disk_usage(r).free,flush=True);out=r/'model_stage_20261106/R19_i3_signnonzero';out.mkdir(parents=True,exist_ok=False)
d=r/'input_alignment_followup_20261106';selection=json.loads((d/'INPUT_BASELINE_I3_SELECTION_20261106.json').read_text());assert selection['engineering_checks_passed'] and selection['validation_guard_passed'];manifest=d/'grouped_features_v1/manifest.json';assert hashlib.sha256(manifest.read_bytes()).hexdigest()==selection['evidence'][a.dataset]['grouped_features_v1/manifest.json']
original=(d/'grouped_control_init_v3/train_control.py').read_text();source=original
replacements={"totalsteps=max(1,12*len(loaders['train']))":"totalsteps=max(1,60*len(loaders['train']))","'epochs':12,'patience':4":"'epochs':60,'patience':15","for epoch in range(1,13):":"for epoch in range(1,61):","if epoch-best_epoch>=4:break":"if epoch-best_epoch>=15:break","'max_epochs':12,'patience':4":"'max_epochs':60,'patience':15","out=r.parent/'input_alignment_followup_20261106/grouped_control_init_v3/validation'/a.modality_control/str(a.seed)":"out=r.parent/'model_stage_20261106/R19_i3_signnonzero/results'/a.modality_control/str(a.seed)","fixed-budget input/fusion diagnostic; not final benchmark result":"shared full-budget model baseline on locked I3; validation selected; test not scored"}
for before,after in replacements.items():assert source.count(before)==1,before;source=source.replace(before,after)
assert source.count("'lr':2e-5")==1
source=source.replace("'lr':2e-5","'lr':1e-5").replace("'text_lr':2e-5","'text_lr':1e-5")
core_path=r/'legacy499/source_musa/bchrpp/full_model.py'
core=core_path.read_text()
marker='            empty = (available & ~dropped).sum(dim=1) == 0'
assert core.count(marker)==1 and "MODALITIES = ('text', 'audio', 'vision')" in core
draw_marker='            dropped = ((torch.rand_like(available, dtype=torch.float) < self.modality_dropout)'
assert core.count(draw_marker)==1
core=core.replace(draw_marker, '            draws = torch.rand_like(available, dtype=torch.float)\n            dropped = ((draws < self.modality_dropout)')
core=core.replace(marker,'            vision_idx = MODALITIES.index("vision")\n            dropped[:, vision_idx] &= draws[:, vision_idx] < .05\n'+marker)
compile(core,'vision_retain_core','exec')
with (out/'vision_retain_core.py').open('x') as f:f.write(core)
assert source.count('from bchrpp.full_model import BCHRPP')==1
source=source.replace('from bchrpp.full_model import BCHRPP','from vision_retain_core import BCHRPP')
loss_import='from losses.bchrpp_loss import compute_loss'
assert source.count(loss_import)==1
source=source.replace(loss_import,'from losses.bchrpp_loss import compute_loss as base_compute_loss')
loss_wrapper='''def compute_loss(prediction, label, aux):
    total, parts = base_compute_loss(prediction, label, aux)
    target = label.reshape_as(prediction)
    positive = (target > 0).to(prediction.dtype)
    non_neutral = (target != 0).to(prediction.dtype)
    per_item = torch.nn.functional.binary_cross_entropy_with_logits(prediction / .5, positive, reduction='none')
    sign_loss = (per_item * non_neutral).sum() / non_neutral.sum().clamp_min(1)
    total = total + .025 * sign_loss
    return total, dict(parts, sign_loss=sign_loss.detach(), total=total.detach())
'''
source=source.replace('assert torch.musa.device_count()==1',loss_wrapper+'\nassert torch.musa.device_count()==1')
assert "evaluate('test')" not in source and "for split in ['train','valid']:" in source;compile(source,'R1_fullbudget','exec')
with (out/'train_model.py').open('x') as f:f.write(source)
plan={'round':'R19_i3_signnonzero','datasets_shared_method':['mosi','sims'],'dataset':a.dataset,'feature_version':'I3_grouped_coverage_v1','feature_manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'core_sha256':hashlib.sha256(core.encode()).hexdigest(),'trainer_sha256':hashlib.sha256(source.encode()).hexdigest(),'previous_trainer_sha256':hashlib.sha256(original.encode()).hexdigest(),'only_changes':list(replacements.items())+[('text_lr',1e-5),('vision_dropout',.05),('sign_aux_weight',.025),('sign_aux_include_zero',False),('sign_logit_temperature',.5)],'rationale':'R18 did not improve validation. Neutral targets put regression at the zero boundary while sign BCE pushes them positive. Test excluding only exact neutral labels from the sign auxiliary, retaining all regression supervision and fixed scoring.','seeds':[222,128,2026],'epochs':60,'patience':15,'batch_size':24,'fusion_lr':3e-4,'text_lr':1e-5,'warmup':.1,'schedule':'cosine to5%baseLR','initialization':'same-seed original-width random core copied; new feature blockszero; CPU RNG restored','full_text_encoder_trainable':True,'test_scoring':False,'larger_datasets_started':False,'next':'collect all6 validation-selected checkpoints and hashes before evaluating tests; no additional automatic tuning branch'}
with (out/'PLAN_LOCK.json').open('x') as f:json.dump(plan,f,indent=2)
env=os.environ.copy();env.update(MUSA_VISIBLE_DEVICES=a.physical_gpu,CUDA_VISIBLE_DEVICES=a.physical_gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
def preflight():
 subprocess.run(['mthreads-gmi'],check=True)
 subprocess.run(['python','-c','import torch,torch_musa;f,t=torch.musa.mem_get_info();assert f>12*1024**3;print("SELECTED_GPU_FREE",f)'],env=env,check=True)
base=['python','-u',str(out/'train_model.py'),'--root',str(r/'legacy499'),'--dataset',a.dataset,'--modality-control','TAV']
preflight()
with (out/'smoke_batch24.log').open('xb') as f:subprocess.run(base+['--seed','222','--smoke'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
results=[]
for seed in [222,128,2026]:
 preflight()
 with (out/f'train_seed{seed}.log').open('xb') as f:subprocess.run(base+['--seed',str(seed)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 path=out/'results/TAV'/str(seed);result=json.loads((path/'VALIDATION_RESULT.json').read_text());assert result['checkpoint_sha256']==hashlib.sha256((path/'best.pt').read_bytes()).hexdigest() and not result['test_scored'];results.append(result)
v=[q['validation_raw_mae'] for q in results];report={'round':'R19_i3_signnonzero','dataset':a.dataset,'seeds':results,'validation_raw_mae_mean':statistics.mean(v),'validation_raw_mae_sample_std':statistics.stdev(v),'test_scored':False,'feature_manifest_sha256':plan['feature_manifest_sha256']}
with (out/'VALIDATION_ROUND_COMPLETE.json').open('x') as f:json.dump(report,f,indent=2)
print('R19_SIGNNONZERO_VALIDATION_COMPLETE',json.dumps(report),flush=True)
