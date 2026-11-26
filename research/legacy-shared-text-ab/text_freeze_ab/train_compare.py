"""Matched full-text frozen/unfrozen experiment on the restored legacy model."""
import argparse,os,sys,json,time,hashlib,pickle,random,math,shutil,io
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--dataset',choices=['sims','mosi'],required=True)
p.add_argument('--physical-gpu',required=True)
p.add_argument('--assets-root',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
os.environ.update(MUSA_VISIBLE_DEVICES=a.physical_gpu,CUDA_VISIBLE_DEVICES=a.physical_gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,torch,torch_musa
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset,DataLoader
from sklearn.metrics import accuracy_score,f1_score
from transformers import AutoModel,AutoTokenizer
from statistics import mean,stdev
from online import OnlineBCHR
from losses.bchrpp_loss import compute_loss
r=Path(__file__).resolve().parent;project=a.assets_root.resolve();out=a.output.resolve()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2)
def report(stage,**values):
 row={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,**values}
 with (out/'STATUS.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
def main():
 out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(4)
 assert torch.musa.device_count()==1
 free,total=torch.musa.mem_get_info();assert free>.85*total
 previous=project/('SIMS_LEGACY_RESTORE_20261125' if a.dataset=='sims' else 'LEGACY_SHARED_MOSI_20261126_v2')
 cache=previous/('caches' if a.dataset=='sims' else 'cache')
 modelpath=previous/'assets/original_roberta_github' if a.dataset=='sims' else project/'AUTHOR_REPRO_MOSI_20261120_v2/assets/roberta-large'
 tokenizerpath=previous/'assets/tokenizer' if a.dataset=='sims' else modelpath
 source=project/'data'/('unaligned_39.pkl' if a.dataset=='sims' else 'aligned_50.pkl')
 expected='c9e20c13ec0454d98bb9c1e520e490c75146bfa2dfeeea78d84de047dbdd442f' if a.dataset=='sims' else 'd3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9'
 assert sha(source)==expected
 with source.open('rb') as f:data=pickle.load(f)
 stats=json.loads((previous/('audit.json' if a.dataset=='sims' else 'TRAIN_ONLY_NORMALIZATION.json')).read_text())
 if a.dataset=='sims':stats=stats['stats_fit_on_train_only']
 tokenizer=AutoTokenizer.from_pretrained(tokenizerpath,local_files_only=True)
 cfg=dict(d=128,output_dim=1,queries=4,text_dim=1024,audio_dim=1024,vision_dim=1024,dropout=.1,topk=4,modality_dropout=.1)
 tokenized={};hashes={}
 def dataset(split):
  columns=[];masks=[]
  for mod in ['text','audio','vision']:
   folder=cache/(('test_'+mod) if split=='test' else mod) if a.dataset=='sims' else cache
   fn=split+'_features.npy' if a.dataset=='sims' else split+'_'+mod+'.npy';mn=split+'_mask.npy' if a.dataset=='sims' else split+'_'+mod+'_mask.npy'
   x=np.load(folder/fn,allow_pickle=False).astype(np.float32);mask=np.load(folder/mn,allow_pickle=False)
   if mod!='text':x=(x-np.asarray(stats[mod]['mean'],np.float32))/np.asarray(stats[mod]['std'],np.float32)
   x[~mask]=0;assert np.isfinite(x).all() and mask.any(1).all()
   columns.append(torch.from_numpy(x));masks.append(torch.from_numpy(mask));hashes[str(folder/fn)]=sha(folder/fn);hashes[str(folder/mn)]=sha(folder/mn)
  tokens=tokenizer([str(t) for t in data[split]['raw_text']],padding='max_length',max_length=columns[0].shape[1],truncation=False,return_tensors='pt')
  assert tokens['input_ids'].shape[:2]==masks[0].shape and np.array_equal(tokens['attention_mask'].numpy().astype(bool),masks[0].numpy())
  tokenized[split]=list(tokens)
  y=torch.tensor(data[split]['regression_labels'],dtype=torch.float32).reshape(-1,1)
  if a.dataset=='sims' and split=='train':targets=torch.from_numpy(np.stack([np.asarray(data[split]['regression_labels_'+m],np.float32).reshape(-1) for m in 'TAV'],1)).unsqueeze(-1)
  else:targets=y[:,None,:].expand(-1,3,-1).clone()
  return TensorDataset(*columns,*masks,y,targets,*tokens.values())
 datasets={s:dataset(s) for s in ['train','valid']}
 def new_model(seed,trainable):
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.musa.manual_seed_all(seed)
  fusion=OnlineBCHR(cfg).float()
  with torch.random.fork_rng(devices=[]):encoder=AutoModel.from_pretrained(modelpath,local_files_only=True,attn_implementation='eager').float()
  for param in encoder.parameters():param.requires_grad_(trainable)
  model=nn.ModuleDict({'fusion_model':fusion,'text_encoder':encoder}).to('musa:0')
  return model
 def batch(raw,split):
  items=[t.to('musa:0') for t in raw];t,au,v,tm,am,vm,y,targets=items[:8]
  tokens=dict(zip(tokenized[split],items[8:]));return t,au,v,{'text':tm,'audio':am,'vision':vm},y,targets,tokens
 def forward(model,b,split,online=False,return_aux=False):
  t,au,v,m,y,targets,tokens=batch(b,split)
  # The same deterministic encoder mode in both arms isolates parameter updates.
  model['text_encoder'].eval()
  if online:
   t=model['text_encoder'](**tokens).last_hidden_state.to(torch.float16).float()
   t=t*tokens['attention_mask'].unsqueeze(-1)
  return model['fusion_model'](t,au,v,masks=m,return_aux=return_aux),y,targets
 @torch.no_grad()
 def evaluate(model,split,online):
  model.eval();ps=[];ys=[]
  for b in DataLoader(datasets[split],batch_size=16,shuffle=False):
   pred,y,_=forward(model,b,split,online);ps.append(pred.cpu());ys.append(y.cpu())
  pred=torch.cat(ps);truth=torch.cat(ys)
  metric=metric_fn(pred,truth);metric['raw_mae']=float((pred.double()-truth.double()).abs().mean());assert all(math.isfinite(v) for v in metric.values())
  return metric,pred.numpy(),truth.numpy()
 sys.path.insert(0,str(project/'legacy499'))
 from cica_metricsTop import MetricsTop
 metric_fn=MetricsTop('regression').getMetics(a.dataset.upper())
 measure=new_model(222,True);buffer=io.BytesIO();torch.save(measure.state_dict(),buffer);fullbytes=buffer.tell();buffer=io.BytesIO();torch.save(measure['fusion_model'].state_dict(),buffer);smallbytes=buffer.tell()
 required=3*fullbytes+3*smallbytes+12*1024**3+32*1024**2
 assert shutil.disk_usage(out).free>=required,(shutil.disk_usage(out).free,required)
 b=next(iter(DataLoader(datasets['train'],batch_size=16)))
 measure.eval()
 with torch.no_grad():
  cached,y,_=forward(measure,b,'train',False);live,_,_=forward(measure,b,'train',True);error=float((cached-live).abs().max())
 assert error<2e-3,error
 report('ENCODER_CACHE_PARITY_PASSED',prediction_max_error=error,full_checkpoint_bytes=fullbytes,required_bytes=required,free_bytes=shutil.disk_usage(out).free)
 # Real full-backward smoke and proof that all used encoder blocks receive gradients.
 measure.train();(prediction,aux),y,targets=forward(measure,b,'train',True,True);loss,_=compute_loss(prediction,y,aux,modality_targets=targets);loss.backward()
 gradients={name:float(param.grad.detach().abs().sum()) for name,param in measure['text_encoder'].named_parameters() if param.grad is not None}
 assert torch.isfinite(loss) and gradients and all(math.isfinite(v) for v in gradients.values())
 assert any('layer.0.' in name and value>0 for name,value in gradients.items()) and any('layer.23.' in name and value>0 for name,value in gradients.items())
 write(out/'FULL_ENCODER_GRADIENT_SMOKE.json',{'loss':float(loss.detach()),'gradient_sums':gradients,'both_first_and_last_blocks_receive_gradient':True})
 del measure,buffer,b;torch.musa.empty_cache()
 plan={'dataset':a.dataset,'arms':['frozen','unfrozen'],'full_encoder_unfrozen':True,'seeds':[222,128,2026],'fusion':cfg,'batch':16,'pretrain_epochs':5,'joint_epochs':20,'patience':5,'fusion_lr':3e-4,'text_lr':1e-5,'normalization':'same train-only stats in both arms','text_train_mode':'eval in both arms; gradients enabled only in unfrozen joint stage','cached_online_output_precision':'FP16 cache representation converted to FP32, same in both arms','native_unimodal_labels':a.dataset=='sims','checkpoint_selection':'minimum validation raw MAE','test_used_for_selection':False,'source_sha256':sha(source),'feature_sha256':hashes,'code_sha256':{str(f.relative_to(r)):sha(f) for f in r.rglob('*.py')}}
 write(out/'PLAN_LOCK.json',plan);results=[]
 for arm in ['frozen','unfrozen']:
  for seed in [222,128,2026]:
   model=new_model(seed,arm=='unfrozen');opt=torch.optim.AdamW([{'params':model['fusion_model'].parameters(),'lr':3e-4},{'params':model['text_encoder'].parameters(),'lr':1e-5}],weight_decay=.01)
   loader=DataLoader(datasets['train'],batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed));folder=out/arm/str(seed);folder.mkdir(parents=True)
   initial_fusion=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model['fusion_model'].state_dict().values())).hexdigest();write(folder/'INITIALIZATION.json',{'fusion_tensor_sha256':initial_fusion,'all_encoder_parameters_trainable':arm=='unfrozen'})
   def epoch(pretrain=False):
    model.train();losses=[]
    for b in loader:
     opt.zero_grad(set_to_none=True);(pred,aux),y,targets=forward(model,b,'train',arm=='unfrozen' and not pretrain,True)
     if pretrain:
      present=aux['modality_available'].to(y).unsqueeze(-1);loss=(F.smooth_l1_loss(aux['unimodal_predictions'],targets,beta=.5,reduction='none')*present).sum()/present.sum().clamp_min(1.)
     else:loss,_=compute_loss(pred,y,aux,modality_targets=targets)
     assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
    return mean(losses)
   for ep in range(1,6):report('PRETRAIN',arm=arm,seed=seed,epoch=ep,loss=epoch(True))
   best=math.inf;bestep=0;state=None
   for ep in range(1,21):
    loss=epoch();metrics,_,_=evaluate(model,'valid',arm=='unfrozen');mae=metrics['raw_mae']
    if mae<best-1e-6:
     best=mae;bestep=ep;state={k:v.detach().cpu().clone() for k,v in (model.state_dict() if arm=='unfrozen' else model['fusion_model'].state_dict()).items()};bestmetrics=metrics
    row={'epoch':ep,'loss':loss,'validation':metrics,'best_epoch':bestep};report('TRAIN',arm=arm,seed=seed,**row)
    with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    if ep-bestep>=5:break
   with (folder/'best.pt').open('xb') as f:torch.save({'state_dict':state,'arm':arm,'config':cfg,'seed':seed,'best_epoch':bestep,'validation':bestmetrics},f)
   row={'arm':arm,'seed':seed,'validation':bestmetrics,'checkpoint_sha256':sha(folder/'best.pt'),'best_epoch':bestep};write(folder/'VALIDATION_RESULT.json',row);results.append(row);report('VALIDATION_COMPLETE',**row)
   del model,opt,state;torch.musa.empty_cache()
 write(out/'PRETEST_LOCK.json',{'six_checkpoints':results,'plan_sha256':sha(out/'PLAN_LOCK.json'),'test_used_for_selection':False})
 report('ALL_SIX_CHECKPOINTS_LOCKED')
 datasets['test']=dataset('test');testrows=[]
 for row in results:
  arm=row['arm'];seed=row['seed'];folder=out/arm/str(seed);assert sha(folder/'best.pt')==row['checkpoint_sha256'];saved=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model=new_model(seed,False)
  if arm=='unfrozen':model.load_state_dict(saved['state_dict'],strict=True)
  else:model['fusion_model'].load_state_dict(saved['state_dict'],strict=True)
  metrics,pred,truth=evaluate(model,'test',arm=='unfrozen')
  with (folder/'TEST_PREDICTIONS.npz').open('xb') as f:np.savez(f,pred=pred,truth=truth,ids=np.asarray(data['test']['id']))
  testrow={'arm':arm,'seed':seed,'test':metrics,'checkpoint_sha256':row['checkpoint_sha256']};write(folder/'TEST_RESULT.json',testrow);testrows.append(testrow);report('TEST_COMPLETE',**testrow);del model,saved;torch.musa.empty_cache()
 summary={'dataset':a.dataset,'split':'test','seeds':[222,128,2026],'individual':testrows,'summary':{arm:{key:{'mean':mean(row['test'][key] for row in testrows if row['arm']==arm),'sample_sd':stdev(row['test'][key] for row in testrows if row['arm']==arm)} for key in testrows[0]['test']} for arm in ['frozen','unfrozen']},'historical_test_nonblind':True}
 write(out/'FINAL_TEST_SUMMARY.json',summary);report('ALL_TESTS_DONE',summary=summary['summary'])
if __name__=='__main__':
 try:main()
 except Exception as e:
  if out.exists():report('FAILED',error=repr(e))
  raise
