"""Standalone raw-media -> I3 features. No earlier experiment caches required."""
import argparse,pathlib,os,json,hashlib,io,zipfile,time
p=argparse.ArgumentParser();p.add_argument('--raw-zip',type=pathlib.Path,required=True);p.add_argument('--metadata',type=pathlib.Path,required=True);p.add_argument('--models',type=pathlib.Path,required=True);p.add_argument('--output',type=pathlib.Path,required=True);p.add_argument('--physical-gpu',required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args()
os.environ.update(MUSA_VISIBLE_DEVICES=a.physical_gpu,CUDA_VISIBLE_DEVICES=a.physical_gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1')
import numpy as np,torch,torch_musa,av
from PIL import Image
from transformers import ClapAudioModelWithProjection,ClapFeatureExtractor,CLIPVisionModelWithProjection,CLIPImageProcessor
try:import zipfile_deflate64
except ImportError:pass
torch.set_num_threads(4);torch.manual_seed(222);torch.musa.manual_seed_all(222);np.random.seed(222);free,total=torch.musa.mem_get_info();assert free>12*1024**3
a.output.mkdir(parents=True,exist_ok=False);meta=json.loads(a.metadata.read_text());rows=meta['samples']
if a.smoke:rows=[q for split in ['train','valid','test'] for q in sorted([q for q in rows if q['split']==split],key=lambda q:hashlib.sha256(q['id'].encode()).hexdigest())[:2]]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
assert digest(a.raw_zip)==meta['raw_zip_sha256'],'raw archive hash mismatch'
for rel,expected in meta['required_AV_model_files_sha256'].items():assert digest(a.models/rel)==expected,rel
print('RAW_AND_MODEL_HASHES_VERIFIED',flush=True)
manifest=dict(meta);manifest.update(samples=rows,code_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),metadata_sha256=hashlib.sha256(a.metadata.read_bytes()).hexdigest(),standalone_raw_media_extraction=True)
with (a.output/'manifest.json').open('x') as f:json.dump(manifest,f,ensure_ascii=False)
am=ClapAudioModelWithProjection.from_pretrained(a.models/'clap_audio_hf',local_files_only=True).eval().float().to('musa:0');ap=ClapFeatureExtractor.from_pretrained(a.models/'clap_audio_hf',local_files_only=True);mel=np.load(a.models/'clap_audio_hf/original_mel_filter.npy');ap.mel_filters=mel;ap.mel_filters_slaney=mel
vm=CLIPVisionModelWithProjection.from_pretrained(a.models/'vision',local_files_only=True,attn_implementation='eager').eval().float().to('musa:0');vp=CLIPImageProcessor.from_pretrained(a.models/'vision',local_files_only=True);fill=tuple(round(x*255) for x in vp.image_mean)
def bounds(n,width,hop):
 if n<=width:return [(0,n)]
 starts=list(range(0,n-width+1,hop))
 if starts[-1]+width<n:starts.append(n-width)
 return [(s,s+width) for s in starts]
def audio_features(wave,bb):
 output=[]
 with torch.inference_mode():
  for i in range(0,len(bb),16):
   batch=ap([wave[s:e] for s,e in bb[i:i+16]],sampling_rate=48000,return_tensors='pt',truncation='rand_trunc',padding='repeatpad');v=am(**{k:v.to('musa:0') for k,v in batch.items()}).audio_embeds;output.append(torch.nn.functional.normalize(v,dim=-1).float().cpu().numpy())
 return np.concatenate(output) if output else np.empty((0,512),np.float32)
def vision_features(images,full=False):
 output=[]
 with torch.inference_mode():
  for i in range(0,len(images),32):
   extra={'do_resize':False,'do_center_crop':False} if full else {};pixels=vp(images=images[i:i+32],return_tensors='pt',**extra).pixel_values.to('musa:0');output.append(vm(pixel_values=pixels).image_embeds.float().cpu().numpy())
 return np.concatenate(output)
records=[];begin=time.monotonic()
with zipfile.ZipFile(a.raw_zip) as z:
 members=z.namelist()
 for index,q in enumerate(rows):
  suffix=q['id'].replace('$_$','/')+'.mp4';hits=[n for n in members if n==suffix or n.endswith('/'+suffix)];assert len(hits)==1;blob=z.read(hits[0]);parts=[];origin=None
  with av.open(io.BytesIO(blob)) as c:
   res=av.AudioResampler(format='fltp',layout='mono',rate=48000)
   for frame in c.decode(audio=0):
    if origin is None:origin=float(frame.time or 0)
    for v in res.resample(frame):parts.append(v.to_ndarray().reshape(-1))
   for v in res.resample(None):parts.append(v.to_ndarray().reshape(-1))
  wave=np.concatenate(parts).astype(np.float32);global_bounds=bounds(len(wave),192000,96000);local_bounds=[x for x in bounds(len(wave),48000,24000) if x not in set(global_bounds)];g=audio_features(wave,global_bounds);local=audio_features(wave,local_bounds);all_audio=np.concatenate([g,local]);iv=np.asarray([[s/48000+origin,e/48000+origin] for s,e in global_bounds+local_bounds],np.float64);old=np.r_[np.arange(len(g),dtype=np.int64),np.full(len(local),-1,np.int64)];order=np.lexsort((iv[:,1],iv[:,0]));iv=iv[order];all_audio=all_audio[order];old=old[order];li=np.flatnonzero(old<0);gi=np.flatnonzero(old>=0)
  if not len(li):li=gi.copy();assert len(li)==1
  centers=iv[li].mean(1);chosen=gi[np.abs(centers[:,None]-iv[gi].mean(1)[None,:]).argmin(1)];audio=np.concatenate([all_audio[li],all_audio[chosen]],1)
  images=[];complete=[];times=[];next_time=-float('inf')
  with av.open(io.BytesIO(blob)) as c:
   rate=float(c.streams.video[0].average_rate or 25)
   for i,frame in enumerate(c.decode(video=0)):
    t=float(frame.time) if frame.time is not None else i/rate
    if t+1e-6<next_time:continue
    image=frame.to_image();w,h=image.size;s=224/max(w,h);nw,nh=max(1,round(w*s)),max(1,round(h*s));canvas=Image.new('RGB',(224,224),fill);canvas.paste(image.resize((nw,nh),Image.Resampling.BICUBIC),((224-nw)//2,(224-nh)//2));images.append(image);complete.append(canvas);times.append(t);next_time=t+.25
  center=vision_features(images);full=vision_features(complete,True);vision=np.concatenate([center,full],1);assert audio.shape[1]==1024 and vision.shape[1]==1536 and np.isfinite(audio).all() and np.isfinite(vision).all()
  with (a.output/(q['id']+'.npz')).open('xb') as f:np.savez_compressed(f,id=np.asarray(q['id']),audio=audio,vision=vision,audio_mask=np.ones(len(audio),bool),vision_mask=np.ones(len(vision),bool),audio_intervals=iv[li],audio_context_intervals=iv[chosen],vision_timestamps=np.asarray(times,np.float64),source_I2_local_audio_indices=li,source_I2_global_audio_indices=chosen)
  records.append({'id':q['id'],'audio_anchors':len(audio),'vision_frames':len(vision),'media_sha256':hashlib.sha256(blob).hexdigest()});print('I3_RAW_EXTRACT',index+1,len(rows),round(time.monotonic()-begin,1),flush=True)
with (a.output/'EXTRACTION_REPORT.json').open('x') as f:json.dump({'samples':len(rows),'records':records,'seconds':time.monotonic()-begin},f,indent=2)
with (a.output/'READY').open('x') as f:f.write(str(len(rows)))
