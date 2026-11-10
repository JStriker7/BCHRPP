"""Score all prespecified seeds after the cross-dataset validation lock. No tuning."""
import argparse,pathlib,json,hashlib,os,subprocess
p=argparse.ArgumentParser();p.add_argument('--root',type=pathlib.Path,required=True);p.add_argument('--dataset',required=True);p.add_argument('--physical-gpu',required=True);a=p.parse_args()
r=a.root;o=r/'model_stage_20261106/R19_i3_signnonzero';dest=o/'locked_test_v1';dest.mkdir(exist_ok=False)
lock=json.loads((r/'R19_TWO_DATASET_PRETEST_LOCK_20261108.json').read_text())
plan=json.loads((o/'PLAN_LOCK.json').read_text());assert hashlib.sha256((o/'train_model.py').read_bytes()).hexdigest()==plan['trainer_sha256']
assert plan==lock['plans'][a.dataset]
assert hashlib.sha256((o/'vision_retain_core.py').read_bytes()).hexdigest()==plan['core_sha256']
for seed in [222,128,2026]:
 d=o/'results/TAV'/str(seed);sha=hashlib.sha256((d/'best.pt').read_bytes()).hexdigest();assert sha==lock[a.dataset][str(seed)]['checkpoint_sha256']==json.loads((d/'selection_lock.json').read_text())['checkpoint_sha256']
source=(o/'train_model.py').read_text().split('if a.smoke:')[0]
assert source.count("for split in ['train','valid']:")==1
source=source.replace("for split in ['train','valid']:","for split in ['test']:")
source=source.replace("ids=[set(s['id'] for s in splits[k].samples) for k in ['train','valid']];assert not(ids[0]&ids[1])", "assert len(splits['test'].samples)>0")
source=source.replace("totalsteps=max(1,60*len(loaders['train']))",'totalsteps=1')
source=source.replace('from vision_retain_core import BCHRPP','sys.path.insert(0,str(Path(__file__).resolve().parents[1]))\nfrom vision_retain_core import BCHRPP')
source+='''
out=r.parent/'model_stage_20261106/R19_i3_signnonzero/locked_test_v1'/str(a.seed);out.mkdir(exist_ok=False)
checkpoint=r.parent/'model_stage_20261106/R19_i3_signnonzero/results/TAV'/str(a.seed)/'best.pt'
saved=torch.load(checkpoint,map_location='cpu',weights_only=False);model.load_state_dict(saved['state_dict'],strict=True)
raw,metrics,pred,truth=evaluate('test')
with (out/'predictions.npz').open('xb') as f:np.savez(f,pred=pred,truth=truth,ids=np.array([q['id'] for q in splits['test'].samples]))
with (out/'TEST_RESULT.json').open('x') as f:json.dump({'dataset':a.dataset,'seed':a.seed,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'raw_mae':raw,'metrics':metrics,'historical_test_seen':True,'independent_blind_test':False,'selection':'all prespecified seeds, minimum validation raw MAE checkpoint, no test-based selection'},f,indent=2)
print('LOCKED_TEST_COMPLETE',a.dataset,a.seed,metrics,flush=True)
'''
compile(source,'locked_test','exec')
with (dest/'evaluate.py').open('x') as f:f.write(source)
env=os.environ.copy();env.update(MUSA_VISIBLE_DEVICES=a.physical_gpu,CUDA_VISIBLE_DEVICES=a.physical_gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
for seed in [222,128,2026]:
 subprocess.run(['mthreads-gmi'],check=True)
 subprocess.run(['python','-c','import torch,torch_musa;assert torch.musa.mem_get_info()[0]>12*1024**3'],env=env,check=True)
 with (dest/f'seed{seed}.log').open('xb') as f:subprocess.run(['python','-u',str(dest/'evaluate.py'),'--root',str(r/'legacy499'),'--dataset',a.dataset,'--seed',str(seed),'--modality-control','TAV'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
