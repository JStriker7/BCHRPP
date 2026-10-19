"""Small train-subset memorization diagnostic. NOT benchmark validation/test."""
import argparse
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from datasets.mosi import MOSIDataset
from bchrpp.full_model import BCHRPP
from losses.bchrpp_loss import compute_loss

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-path',type=Path,default=ROOT/'data'/'MOSI'/'Processed'/'unaligned_50.pkl')
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--hidden-dim',type=int,default=64)
    p.add_argument('--lr',type=float,default=.001)
    p.add_argument('--seed',type=int,default=222)
    args=p.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    ds=MOSIDataset(args.data_path,split='train')
    subset=Subset(ds,range(min(len(ds),args.samples)))
    item=subset[0]
    cfg=dict(d=args.hidden_dim,queries=2,modality_dropout=0.,
             **{m+'_dim':item[m].shape[-1] for m in ('text','audio','vision')})
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=BCHRPP(**cfg).to(device)
    optim=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=0.)
    loader=DataLoader(subset,batch_size=args.batch_size,shuffle=True)
    first,last=None,None
    for epoch in range(1,args.epochs+1):
        model.train()
        err,n=0.,0
        for batch in loader:
            inputs=[batch[m].to(device) for m in ('text','audio','vision')]
            masks={m:batch[m+'_mask'].to(device) for m in ('text','audio','vision')}
            y=batch['label'].to(device)
            optim.zero_grad()
            output,aux=model(*inputs,masks=masks,return_aux=True)
            loss,_=compute_loss(output,y,aux)
            if not torch.isfinite(loss):raise RuntimeError('nonfinite diagnostic loss')
            loss.backward();optim.step()
            err+=float(loss.detach())*len(y);n+=len(y)
        last=err/n
        if first is None:first=last
        if epoch==1 or epoch%10==0 or epoch==args.epochs:
            print(f'epoch={epoch:3d} train_objective={last:.5f}')
    print(f'finished training diagnostic: first={first:.6f} last={last:.6f}; ratio={last/first:.4f}')
    print('Interpretation: memorization on a tiny subset is not generalization.')

if __name__=='__main__':main()
