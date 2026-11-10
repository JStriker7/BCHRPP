"""Historical modality/full policy transferred to four datasets; MUSA FP32."""
import argparse,hashlib,json,math,random,sys,time
from pathlib import Path
import numpy as np,torch,torch_musa
from torch.utils.data import DataLoader
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--dataset',required=True);p.add_argument('--seed',type=int,choices=[222,128,2026],required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--batch-size',type=int,default=24);p.add_argument('--modality-control',choices=['T','TAV'],required=True);a=p.parse_args();r=a.root.resolve();sys.path.insert(0,str(r/'source_musa'));sys.path.insert(0,str(r))
from vision_retain_core import BCHRPP
from bchrpp.mosi_online_text import MOSIOnlineText
from datasets.mosi_text_online import MOSIOnlineTextSplit,load_word_tokenizer
from losses.bchrpp_loss import compute_loss as base_compute_loss
from cica_metricsTop import MetricsTop
from simsv2_metrics import simsv2_metrics
from transformers import AutoModel
def compute_loss(prediction, label, aux):
    total, parts = base_compute_loss(prediction, label, aux)
    target = label.reshape_as(prediction)
    positive = (target > 0).to(prediction.dtype)
    non_neutral = (target != 0).to(prediction.dtype)
    per_item = torch.nn.functional.binary_cross_entropy_with_logits(prediction / .5, positive, reduction='none')
    sign_loss = (per_item * non_neutral).sum() / non_neutral.sum().clamp_min(1)
    total = total + .025 * sign_loss
    return total, dict(parts, sign_loss=sign_loss.detach(), total=total.detach())

assert torch.musa.device_count()==1
free,total=torch.musa.mem_get_info();assert free>12*1024**3,(free,total)
torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.musa.manual_seed_all(a.seed)
features=r.parent/'input_alignment_followup_20261106/grouped_features_v1';assert (features/'READY').exists();manifest=json.loads((features/'manifest.json').read_text());rows=manifest['samples'];tok=load_word_tokenizer(r/'models/text_v2');assert tok.vocab_size>20000
splits={}
for split in ['train','valid']:
 samples=[]
 for row in rows:
  if row['split']!=split:continue
  with np.load(features/(row['id']+'.npz'),allow_pickle=False) as x:
   assert str(x['id'])==row['id'];n=len(row['words'])
   samples.append({'id':row['id'],'label':row['label'],'words':row['words'],'text':np.zeros((n,1024),np.float32),'text_valid':np.ones(n,bool),'audio':x['audio'].astype(np.float32),'audio_valid':x['audio_mask'].copy(),'visual':x['vision'].astype(np.float32),'visual_valid':x['vision_mask'].copy()})
 splits[split]=MOSIOnlineTextSplit(samples,tok)
ids=[set(s['id'] for s in splits[k].samples) for k in ['train','valid']];assert not(ids[0]&ids[1])
loaders={k:DataLoader(v,batch_size=a.batch_size,shuffle=k=='train',collate_fn=v.collate,num_workers=0,generator=torch.Generator().manual_seed(a.seed)) for k,v in splits.items()}
encoder=AutoModel.from_pretrained(r/'models/text_v2',local_files_only=True,attn_implementation='eager').float();assert encoder.config.hidden_size==1024
cfg=dict(d=128,output_dim=1,queries=4,text_dim=1024,audio_dim=1024,vision_dim=1536,dropout=.1,topk=4,temporal=True,sparse_graph=True,reliability=True,complementarity=True,refinement=True,modality_dropout=.1,routing_temperature=1.)
reference_cfg=dict(cfg,audio_dim=512,vision_dim=768)
reference=BCHRPP(**reference_cfg)
reference_cpu_rng=torch.get_rng_state().clone()
fusion=BCHRPP(**cfg)
reference_state=reference.state_dict();expanded_state=fusion.state_dict();copied=[]
for key,value in reference_state.items():
 if key=='audio_projection.weight':
  expanded_state[key]=torch.zeros_like(expanded_state[key]);expanded_state[key][:,512:]=value
 elif key=='vision_projection.weight':
  expanded_state[key]=torch.zeros_like(expanded_state[key]);expanded_state[key][:,:768]=value
 else:
  assert value.shape==expanded_state[key].shape;expanded_state[key]=value.clone();copied.append(key)
fusion.load_state_dict(expanded_state,strict=True)
assert all(torch.equal(fusion.state_dict()[k],reference_state[k]) for k in copied)
first=next(q for q in rows if q['split']=='train')
with np.load(features/(first['id']+'.npz')) as check:
 au=torch.from_numpy(check['audio'][:2]);vi=torch.from_numpy(check['vision'][:2])
 with torch.no_grad():
  adiff=float((fusion.audio_projection(au)-reference.audio_projection(au[:,512:])).abs().max());vdiff=float((fusion.vision_projection(vi)-reference.vision_projection(vi[:,:768])).abs().max());assert max(adiff,vdiff)<1e-5
init_evidence={'all_other_core_tensors_bitwise_identical_to_original_seeded_initialization':True,'matching_tensor_count':len(copied),'reference_random_weights_only':True,'audio_new_local_block_zero':bool((fusion.audio_projection.weight[:,:512]==0).all()),'vision_new_fullfield_block_zero':bool((fusion.vision_projection.weight[:,768:]==0).all()),'actual_feature_initial_projection_max_error':{'audio':adiff,'vision':vdiff},'CPU_rng_restored_to_after_original_width_construction':True}
torch.set_rng_state(reference_cpu_rng)
del reference,reference_state,expanded_state,au,vi
model=MOSIOnlineText(encoder,fusion,train_text=True).float().to('musa:0');opt=torch.optim.AdamW([{'params':model.fusion.parameters(),'lr':3e-4},{'params':model.encoder.parameters(),'lr':1e-5}],weight_decay=.01)
totalsteps=max(1,60*len(loaders['train']));warmup=int(totalsteps*.1)
def scale(step):
 if step<warmup:return max((step+1)/max(warmup,1),.001)
 t=min(1.,(step-warmup)/max(totalsteps-warmup,1));return .05+.95*.5*(1+math.cos(math.pi*t))
scheduler=torch.optim.lr_scheduler.LambdaLR(opt,scale)
metric=simsv2_metrics if a.dataset=='simsv2' else MetricsTop('regression').getMetics(a.dataset.upper())
def batch(raw):
 text,audio,vision=[raw[k].to('musa:0') for k in ['text','audio','vision']]
 masks={k:raw[k+'_mask'].to('musa:0') for k in ['text','audio','vision']}
 if a.modality_control=='T':
  masks['audio']=torch.zeros_like(masks['audio']);masks['vision']=torch.zeros_like(masks['vision'])
 tokens=({k:raw['text_'+k].to('musa:0') for k in ['input_ids','attention_mask','token_type_ids'] if 'text_'+k in raw},raw['text_word_ids'].to('musa:0'))
 return text,audio,vision,masks,tokens,raw['label'].to('musa:0')
def train_epoch(limit=None):
 model.train();losses=[]
 for i,raw in enumerate(loaders['train']):
  t,au,v,m,tt,y=batch(raw);opt.zero_grad(set_to_none=True);pred,aux=model(t,au,v,masks=m,text_tokens=tt,return_aux=True);loss,_=compute_loss(pred,y,aux);assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();scheduler.step();losses.append(float(loss.detach()))
  if limit and i+1>=limit:break
 return float(np.mean(losses))
@torch.no_grad()
def evaluate(split):
 model.eval();ps=[];ys=[]
 for raw in loaders[split]:
  t,au,v,m,tt,y=batch(raw);ps.append(model(t,au,v,masks=m,text_tokens=tt).float().cpu());ys.append(y.float().cpu())
 pred,truth=torch.cat(ps),torch.cat(ys);raw=float((pred.double()-truth.double()).abs().mean());metrics=metric(pred,truth);assert math.isfinite(raw);return raw,metrics,pred.numpy(),truth.numpy()
if a.smoke:print(json.dumps({'legacy499_smoke_passed':True,'loss':train_epoch(2),'physical_batch':a.batch_size}),flush=True);sys.exit(0)
out=r.parent/'model_stage_20261106/R19_i3_signnonzero/results'/a.modality_control/str(a.seed);out.mkdir(parents=True,exist_ok=False)
codehash=hashlib.sha256(Path(__file__).read_bytes()+(r/'source_musa/bchrpp/full_model.py').read_bytes()+(r/'source_musa/bchrpp/mosi_online_text.py').read_bytes()+(r/'source_musa/losses/bchrpp_loss.py').read_bytes()).hexdigest()
record={'input_width_initialization_control':init_evidence,'dataset':a.dataset,'seed':a.seed,'source_commit':'43eb9c9a3033270354fe9db7133ed206b3f32de1','source_sha256':codehash,'features_manifest_sha256':hashlib.sha256((features/'manifest.json').read_bytes()).hexdigest(),'configuration':cfg,'epochs':60,'patience':15,'batch_size':a.batch_size,'fusion_lr':3e-4,'text_lr':1e-5,'warmup_ratio':.1,'schedule':'cosine to5% base lr','precision':'MUSA FP32, historical CUDA AMP differs','full_text_encoder_trainable':True,'normalization':'none beyond original encoder outputs','legacy_exact_reproduction':False,'test_used_for_selection':False}
with (out/'manifest.json').open('x') as f:json.dump(record,f,indent=2)
best=float('inf');best_epoch=0;state=None;best_metrics=None
for epoch in range(1,61):
 start=time.monotonic();loss=train_epoch();mae,metrics,_,_=evaluate('valid')
 if mae<best:
  best=mae;best_epoch=epoch;best_metrics=metrics;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
 log={'epoch':epoch,'loss':loss,'validation_raw_mae':mae,'validation':metrics,'best_epoch':best_epoch,'seconds':round(time.monotonic()-start,1)};print(json.dumps(log),flush=True)
 with (out/'epochs.jsonl').open('a') as f:f.write(json.dumps(log)+'\n')
 if epoch-best_epoch>=15:break
with (out/'best.pt').open('xb') as f:torch.save({'state_dict':state,'manifest':record,'best_epoch':best_epoch,'validation_raw_mae':best},f)
sha=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest()
with (out/'selection_lock.json').open('x') as f:json.dump({'checkpoint_sha256':sha,'validation_raw_mae':best,'best_epoch':best_epoch,'selection':'minimum validation raw MAE'},f,indent=2)

with (out/'VALIDATION_RESULT.json').open('x') as f:
 json.dump({'dataset':a.dataset,'control':a.modality_control,'seed':a.seed,'validation_raw_mae':best,'best_epoch':best_epoch,'checkpoint_sha256':sha,'max_epochs':60,'patience':15,'batch_size':a.batch_size,'test_scored':False,'purpose':'shared full-budget model baseline on locked I3; validation selected; test not scored'},f,indent=2)
print('VALIDATION_CONTROL_COMPLETE',a.modality_control,best,flush=True)
