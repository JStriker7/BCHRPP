"""Config smoke entry point (real training: train_mosi.py/train_simsv2.py)."""
import argparse
import json
import random
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
import yaml
from bchrpp.full_model import BCHRPP
from losses.bchrpp_loss import compute_loss


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--smoke-test',action='store_true')
    args=parser.parse_args()
    conf=yaml.safe_load(args.config.read_text(encoding='utf8'))
    torch.manual_seed(int(conf.get('seed',222)))
    random.seed(int(conf.get('seed',222)))
    if not args.smoke_test:
        raise SystemExit('Use scripts/train_mosi.py or scripts/train_simsv2.py for real benchmark training.')
    dims={'text_dim':12,'audio_dim':5,'vision_dim':8}
    task=conf.get('task','regression')
    out_dim=int(conf.get('num_classes',7)) if task=='classification' else 1
    model=BCHRPP(d=int(conf['hidden_dim']),queries=int(conf.get('num_queries',4)),output_dim=out_dim,**dims)
    n=min(int(conf['batch_size']),3)
    x={m:torch.randn(n,6,dims[m+'_dim']) for m in ('text','audio','vision')}
    mask={m:torch.ones(n,6,dtype=torch.bool) for m in x}
    mask['audio'][:,3:]=False
    if n>1:mask['vision'][1]=False
    y=torch.randint(0,out_dim,(n,)) if task=='classification' else torch.randn(n,1)
    model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=float(conf['lr']))
    optimizer.zero_grad()
    predictions,aux=model(x['text'],x['audio'],x['vision'],masks=mask,return_aux=True)
    loss,terms=compute_loss(predictions,y,aux,task=task)
    loss.backward();optimizer.step()
    print(json.dumps({'status':'passed','dataset':args.dataset,'task':task,'loss':loss.item(),
                      'shape':list(predictions.shape),'params':sum(p.numel() for p in model.parameters()),
                      'loss_terms':{k:float(v) for k,v in terms.items()}},indent=2))


if __name__=='__main__':main()
