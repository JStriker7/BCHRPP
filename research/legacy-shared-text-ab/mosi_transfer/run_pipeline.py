"""Legacy SIMS method transferred to MOSI; frozen encoders, unchanged fusion/loss."""
import argparse,os,sys,json,time,hashlib,pickle,subprocess,shutil,random,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--physical-gpu',required=True);a=p.parse_args()
os.environ.update(MUSA_VISIBLE_DEVICES=a.physical_gpu,CUDA_VISIBLE_DEVICES=a.physical_gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
r=Path(__file__).resolve().parent;project=r.parent
def report(stage,**kw):
 v={'time':time.strftime('%Y-%m-%d %H:%M:%S'),'stage':stage,**kw}
 with (r/'STATUS.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
 print(json.dumps(v),flush=True)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def write(path,v):
 with path.open('x') as f:json.dump(v,f,indent=2)
def acquire_hubert():
 verified=r.parent/'LEGACY_SHARED_MOSI_20261126/hubert'
 proof=json.loads((verified/'FILES_VERIFIED.json').read_text())
 assert all(sha(verified/name)==checksum for name,checksum in proof.items())
 return verified
 import requests
 folder=r/'hubert';folder.mkdir(exist_ok=False)
 repo='facebook/hubert-large-ls960-ft'
 info=requests.get('https://hf-mirror.com/api/models/'+repo,timeout=60);info.raise_for_status();revision=info.json()['sha']
 write(folder/'SOURCE_LOCK.json',{'repo':repo,'revision':revision})
 for name in ['config.json','preprocessor_config.json','pytorch_model.bin']:
  path=folder/name;url=f'https://hf-mirror.com/{repo}/resolve/{revision}/{name}'
  for attempt in range(10):
   offset=path.stat().st_size if path.exists() else 0
   try:
    report('DOWNLOAD_HUBERT',file=name,offset=offset,attempt=attempt)
    with requests.get(url,headers={'Range':f'bytes={offset}-'} if offset else {},stream=True,timeout=(30,90)) as response:
     response.raise_for_status()
     if offset:assert response.status_code==206 and response.headers.get('Content-Range','').startswith(f'bytes {offset}-')
     with path.open('ab' if offset else 'xb') as f:
      for block in response.iter_content(256*1024):
       if block:f.write(block)
    break
   except Exception as e:
    report('DOWNLOAD_RETRY',error=repr(e)[:180]);time.sleep(5)
  else:raise RuntimeError('download failed '+name)
 write(folder/'FILES_VERIFIED.json',{name:sha(folder/name) for name in ['config.json','preprocessor_config.json','pytorch_model.bin']})
 return folder
def main():
 import numpy as np,torch,torch_musa,cv2,imageio_ffmpeg
 from transformers import AutoTokenizer,AutoModel,HubertModel,Wav2Vec2FeatureExtractor,CLIPModel,CLIPImageProcessor
 from torch.utils.data import TensorDataset,DataLoader
 import torch.nn.functional as F
 from online import OnlineBCHR
 from losses.bchrpp_loss import compute_loss
 # Use the same audited MOSI scorer as R19 for the primary comparison.
 sys.path.insert(0,str(project/'legacy499'))
 from cica_metricsTop import MetricsTop
 torch.set_num_threads(4)
 assert torch.musa.device_count()==1
 free,total=torch.musa.mem_get_info();assert free>.85*total
 assert shutil.disk_usage(r).free>12*1024**3+4*1024**3
 source=project/'data/aligned_50.pkl';assert sha(source)=='d3994fd25681f9c7ad6e9c6596a6fe9b4beb85ff7d478ba978b124139002e5f9'
 with source.open('rb') as f:data=pickle.load(f)
 counts={'train':1284,'valid':229,'test':686}
 for split,n in counts.items():assert len(data[split]['id'])==n and len(set(data[split]['id']))==n
 assert not(set(data['train']['id'])&set(data['valid']['id'])) and not(set(data['test']['id'])&(set(data['train']['id'])|set(data['valid']['id'])))
 textpath=project/'AUTHOR_REPRO_MOSI_20261120_v2/assets/roberta-large';clipath=project/'legacy499/models/vision'
 assert sha(clipath/'model.safetensors')=='a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb'
 raw=project/'AUTHOR_REPRO_MOSI_20261120_v2/assets/raw/Raw'
 for split in counts:
  for sample in data[split]['id']:assert (raw/(sample.replace('$_$','/')+'.mp4')).is_file(),sample
 report('PREFLIGHT_PASSED',free_gpu=free,free_disk=shutil.disk_usage(r).free)
 hubert=acquire_hubert()
 tokenizer=AutoTokenizer.from_pretrained(textpath,local_files_only=True)
 longest=max(len(tokenizer.encode(t)) for split in ['train','valid'] for t in data[split]['raw_text'])
 assert longest<=512
 maxtext=max(64,math.ceil(longest/8)*8)
 cfg=dict(d=128,output_dim=1,queries=4,text_dim=1024,audio_dim=1024,vision_dim=1024,dropout=.1,topk=4,modality_dropout=.1)
 write(r/'CONFIG_LOCK.json',{'source_sha256':sha(source),'model_config':cfg,'max_text_tokens':maxtext,'hubert_source':json.loads((hubert/'SOURCE_LOCK.json').read_text()),'model_asset_sha':{str(f):sha(f) for folder in [textpath,clipath,hubert] for f in folder.glob('*') if f.is_file() and f.suffix in ['.bin','.safetensors','.json']},'code_sha':{str(f.relative_to(r)):sha(f) for f in r.rglob('*.py')},'test_used_for_selection':False})
 cache=r/'cache';cache.mkdir();stored={}
 def extract(splits):
  for split in splits:
   assert max(len(tokenizer.encode(t)) for t in data[split]['raw_text'])<=maxtext, 'Text longer than prelocked cap; no truncation allowed'
  models=[('text',AutoModel.from_pretrained(textpath,local_files_only=True,attn_implementation='eager').float().eval().to('musa:0'))]
  for mod,model in models:
   for split in splits:
    n=counts[split];x=np.zeros((n,maxtext,1024),np.float16);mask=np.zeros((n,maxtext),bool)
    for i in range(0,n,8):
     tokens=tokenizer([str(v) for v in data[split]['raw_text'][i:i+8]],padding='max_length',max_length=maxtext,truncation=False,return_tensors='pt')
     with torch.inference_mode():z=model(**{k:v.to('musa:0') for k,v in tokens.items()}).last_hidden_state.float().cpu().numpy()
     valid=tokens['attention_mask'].numpy().astype(bool);z[~valid]=0;x[i:i+len(z)]=z;mask[i:i+len(z)]=valid
    assert np.isfinite(x).all();np.save(cache/(split+'_text.npy'),x);np.save(cache/(split+'_text_mask.npy'),mask);report('EXTRACTED',split=split,modality='text',n=n)
   del model;models.clear();torch.musa.empty_cache()
  for mod in ['audio','vision']:
   if mod=='audio':processor=Wav2Vec2FeatureExtractor.from_pretrained(hubert,local_files_only=True);model=HubertModel.from_pretrained(hubert,local_files_only=True,attn_implementation='eager').float().eval().to('musa:0');length=128
   else:processor=CLIPImageProcessor.from_pretrained(clipath,local_files_only=True);model=CLIPModel.from_pretrained(clipath,local_files_only=True,attn_implementation='eager').float().eval().to('musa:0');length=8
   for split in splits:
    n=counts[split];x=np.zeros((n,length,1024),np.float16);mask=np.zeros((n,length),bool)
    for i,sample in enumerate(data[split]['id']):
     path=raw/(sample.replace('$_$','/')+'.mp4')
     if mod=='audio':
      audio=subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-i',str(path),'-vn','-ac','1','-ar','16000','-f','f32le','pipe:1'],capture_output=True,check=True).stdout
      wave=np.frombuffer(audio,dtype=np.float32).copy();assert len(wave)>0
      inputs=processor(wave,sampling_rate=16000,return_tensors='pt')
      with torch.inference_mode():z=model(**{k:v.to('musa:0') for k,v in inputs.items()}).last_hidden_state[0]
      indices=torch.linspace(0,len(z)-1,min(len(z),128),device=z.device).round().long();z=z.index_select(0,indices).float().cpu().numpy()
     else:
      cap=cv2.VideoCapture(str(path));totalframes=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));assert totalframes>0
      desired=np.rint(np.linspace(0,totalframes-1,8)).astype(int);frames=[];j=0;k=0
      while k<8:
       ok,frame=cap.read()
       if not ok:break
       while k<8 and desired[k]==j:frames.append(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB));k+=1
       j+=1
      cap.release();assert frames
      while len(frames)<8:frames.append(frames[-1])
      inputs=processor(images=frames,return_tensors='pt')
      with torch.inference_mode():z=model.vision_model(pixel_values=inputs['pixel_values'].to('musa:0')).pooler_output.float().cpu().numpy()
     assert np.isfinite(z).all() and z.shape[1]==1024;x[i,:len(z)]=z;mask[i,:len(z)]=True
     if (i+1)%200==0:report('EXTRACT_PROGRESS',split=split,modality=mod,done=i+1,n=n)
    np.save(cache/(split+'_'+mod+'.npy'),x);np.save(cache/(split+'_'+mod+'_mask.npy'),mask);report('EXTRACTED',split=split,modality=mod,n=n)
   del model;torch.musa.empty_cache()
 extract(['train','valid'])
 stats={}
 for mod in ['audio','vision']:
  x=np.load(cache/('train_'+mod+'.npy')).astype(np.float32);mask=np.load(cache/('train_'+mod+'_mask.npy'));valid=x[mask]
  mu=valid.mean(0,dtype=np.float64).astype(np.float32);sd=valid.std(0,dtype=np.float64).astype(np.float32);sd=np.where(sd>1e-3,sd,1.)
  stats[mod]={'mean':mu.tolist(),'std':sd.tolist()}
 write(r/'TRAIN_ONLY_NORMALIZATION.json',stats)
 def dataset(split):
  cols=[];masks=[]
  for mod in ['text','audio','vision']:
   x=np.load(cache/(split+'_'+mod+'.npy')).astype(np.float32);mask=np.load(cache/(split+'_'+mod+'_mask.npy'))
   if mod!='text':x=(x-np.array(stats[mod]['mean'],np.float32))/np.array(stats[mod]['std'],np.float32)
   x[~mask]=0;assert np.isfinite(x).all() and mask.any(1).all();cols.append(torch.from_numpy(x));masks.append(torch.from_numpy(mask))
  return TensorDataset(*cols,*masks,torch.tensor(data[split]['regression_labels'],dtype=torch.float32).reshape(-1,1))
 datasets={s:dataset(s) for s in ['train','valid']};metric=MetricsTop('regression').getMetics('MOSI')
 def move(raw):
  t,au,v,tm,am,vm,y=[v.to('musa:0') for v in raw];return t,au,v,{'text':tm,'audio':am,'vision':vm},y
 @torch.no_grad()
 def evaluate(model,split):
  model.eval();ps=[];ys=[]
  for rawbatch in DataLoader(datasets[split],batch_size=16,shuffle=False):
   t,au,v,m,y=move(rawbatch);ps.append(model(t,au,v,masks=m).cpu());ys.append(y.cpu())
  pred=torch.cat(ps);truth=torch.cat(ys);mae=float((pred.double()-truth.double()).abs().mean());assert math.isfinite(mae)
  return mae,metric(pred,truth),pred.numpy(),truth.numpy()
 runs=r/'runs';runs.mkdir();selected={}
 for seed in [222,128,2026]:
  random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.musa.manual_seed_all(seed)
  model=OnlineBCHR(cfg).float().to('musa:0');opt=torch.optim.AdamW([{'params':model.fusion.parameters(),'lr':3e-4},{'params':model.text_skip.parameters(),'lr':3e-4}],weight_decay=.01)
  loader=DataLoader(datasets['train'],batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed));folder=runs/str(seed);folder.mkdir()
  def epoch(pretrain=False,limit=None):
   model.train();losses=[]
   for i,batch in enumerate(loader):
    t,au,v,m,y=move(batch);opt.zero_grad(set_to_none=True);prediction,aux=model(t,au,v,masks=m,return_aux=True);targets=y[:,None,:].expand_as(aux['unimodal_predictions'])
    if pretrain:
     present=aux['modality_available'].to(y).unsqueeze(-1);loss=(F.smooth_l1_loss(aux['unimodal_predictions'],targets,beta=.5,reduction='none')*present).sum()/present.sum().clamp_min(1.)
    else:loss,_=compute_loss(prediction,y,aux,modality_targets=targets)
    assert torch.isfinite(loss);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.);assert torch.isfinite(norm);opt.step();losses.append(float(loss.detach()))
    if limit and i+1>=limit:break
   return sum(losses)/len(losses)
  # Smoke on a separate seeded model, then restart the formal initialization.
  if seed==222:
   for pre in [True,False]:epoch(pre,2)
   report('REAL_BATCH16_SMOKE_PASSED')
   random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.musa.manual_seed_all(seed);model=OnlineBCHR(cfg).float().to('musa:0');opt=torch.optim.AdamW([{'params':model.fusion.parameters(),'lr':3e-4},{'params':model.text_skip.parameters(),'lr':3e-4}],weight_decay=.01);loader=DataLoader(datasets['train'],batch_size=16,shuffle=True,generator=torch.Generator().manual_seed(seed))
  for ep in range(1,6):report('PRETRAIN',seed=seed,epoch=ep,loss=epoch(True))
  best=math.inf;bestep=0;state=None
  for ep in range(1,21):
   loss=epoch();mae,metrics,_,_=evaluate(model,'valid')
   if mae<best-1e-6:best=mae;bestep=ep;state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
   row={'epoch':ep,'loss':loss,'validation_raw_mae':mae,'validation':metrics,'best_epoch':bestep};report('TRAIN',seed=seed,**row)
   with (folder/'epochs.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
   if ep-bestep>=5:break
  with (folder/'best.pt').open('xb') as f:torch.save({'state_dict':state,'config':cfg,'best_epoch':bestep,'validation_raw_mae':best},f)
  selected[seed]={'sha256':sha(folder/'best.pt'),'best_epoch':bestep,'validation_raw_mae':best};write(folder/'SELECTION.json',selected[seed]);del model,opt,state;torch.musa.empty_cache()
 write(r/'PRETEST_LOCK.json',{'selected':selected,'config_sha256':sha(r/'CONFIG_LOCK.json'),'feature_sha256':{f.name:sha(f) for f in cache.glob('*.npy')},'test_used_for_selection':False})
 report('THREE_SEED_VALIDATION_DONE')
 extract(['test']);datasets['test']=dataset('test');rows=[]
 for seed in [222,128,2026]:
  folder=runs/str(seed);assert sha(folder/'best.pt')==selected[seed]['sha256'];checkpoint=torch.load(folder/'best.pt',map_location='cpu',weights_only=False);model=OnlineBCHR(cfg).float().to('musa:0');model.load_state_dict(checkpoint['state_dict'],strict=True)
  mae,metrics,pred,truth=evaluate(model,'test');metrics['raw_mae']=mae
  with (folder/'test_predictions.npz').open('xb') as f:np.savez(f,pred=pred,truth=truth,ids=np.array(data['test']['id']))
  rows.append({'seed':seed,'test':metrics});write(folder/'TEST_RESULT.json',rows[-1]);report('TEST_COMPLETE',**rows[-1]);del model,checkpoint;torch.musa.empty_cache()
 from statistics import mean,stdev
 summary={'dataset':'mosi','split':'test','seeds':[222,128,2026],'individual':rows,'means':{k:mean(x['test'][k] for x in rows) for k in rows[0]['test']},'sample_sd':{k:stdev(x['test'][k] for x in rows) for k in rows[0]['test']},'historical_test_nonblind':True}
 write(r/'FINAL_SUMMARY.json',summary);report('ALL_TESTS_DONE',means=summary['means'])
if __name__=='__main__':
 try:main()
 except Exception as e:report('FAILED',error=repr(e));raise
