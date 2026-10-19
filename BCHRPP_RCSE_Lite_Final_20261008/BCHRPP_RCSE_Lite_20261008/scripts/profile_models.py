"""Reproduce same-device parameter / latency comparison (no accuracy claims)."""
import argparse
import statistics
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import torch
from bchrpp.full_model import BCHRPP as Lite
from bchrpp.enhanced_v1 import BCHRPP as Enhanced
from bchrpp.legacy_model import BCHRPP as Legacy

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dims',type=int,default=128)
    p.add_argument('--queries',type=int,default=4)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--time-steps',type=int,default=50)
    p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--repeat',type=int,default=40)
    p.add_argument('--threads',type=int,default=1)
    args=p.parse_args()
    torch.set_num_threads(args.threads)
    xs=[torch.randn(args.batch,args.time_steps,v) for v in (768,5,20)]
    masks={k:torch.ones(args.batch,args.time_steps,dtype=torch.bool) for k in ('text','audio','vision')}
    print('device=cpu','torch='+torch.__version__,'threads='+str(args.threads))
    for name,cls in [('legacy',Legacy),('enhanced_v1',Enhanced),('lite',Lite)]:
        kwargs=dict(d=args.dims,queries=args.queries)
        if name != 'legacy':kwargs.update(dropout=0.,modality_dropout=0.)
        model=cls(**kwargs).eval().cpu()
        with torch.no_grad():
            for _ in range(args.warmup):model(*xs,masks=masks)
            durations=[]
            for _ in range(args.repeat):
                start=time.perf_counter()
                model(*xs,masks=masks)
                durations.append((time.perf_counter()-start)*1000)
        print(f'{name}: params={sum(v.numel() for v in model.parameters())}; '
              f'mean_ms={statistics.mean(durations):.3f}; median_ms={statistics.median(durations):.3f}')

if __name__=='__main__':main()
