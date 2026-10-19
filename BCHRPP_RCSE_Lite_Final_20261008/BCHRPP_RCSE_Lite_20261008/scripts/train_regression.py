"""Reproducible MOSI / CH-SIMSv2 benchmark runner for BCHR++ variants.

Only pre-extracted features matching the provided loaders are supported.
Checkpoint selected by validation MAE; test is evaluated after selection.
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets.mosi import MOSIDataset
from datasets.simsv2 import load_simsv2_datasets
from metrics.evaluation import mosi_metrics, simsv2_metrics
from bchrpp.full_model import BCHRPP as LiteBCHRPP
from bchrpp.enhanced_v1 import BCHRPP as EnhancedBCHRPP
from bchrpp.legacy_model import BCHRPP as LegacyBCHRPP
from losses.bchrpp_loss import compute_loss, task_loss
from losses.enhanced_v1_loss import compute_loss as old_enhanced_loss

MODEL_TYPES = {'lite': LiteBCHRPP, 'enhanced_v1': EnhancedBCHRPP, 'legacy': LegacyBCHRPP}
VERSION_TYPES = {'bchrpp_rcse_lite_v1': 'lite', 'bchrpp_enhanced_v1': 'enhanced_v1', 'bchrpp_legacy':'legacy'}


def parser_for(dataset):
    p = argparse.ArgumentParser(description=f'BCHR++ RCSE-Lite / reference {dataset} trainer')
    p.add_argument('--model', choices=MODEL_TYPES, default='lite')
    p.add_argument('--legacy-baseline', action='store_true', help='alias for --model legacy')
    p.add_argument('--data-path', type=Path, default=ROOT/'data'/('MOSI' if dataset=='mosi' else 'SIMSv2')/'Processed'/('unaligned_50.pkl' if dataset=='mosi' else 'unaligned.pkl'))
    p.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints'/f'{dataset}_rcse_lite_best.pt')
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--batch-size', type=int, default=24 if dataset=='mosi' else 16)
    p.add_argument('--hidden-dim', type=int, default=128)
    p.add_argument('--queries', type=int, default=4)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight-decay', type=float, default=.01)
    p.add_argument('--warmup-ratio', type=float, default=.1)
    p.add_argument('--dropout', type=float, default=.1)
    p.add_argument('--modality-dropout', type=float, default=.1)
    p.add_argument('--topk', type=int, default=4)
    p.add_argument('--routing-temperature', type=float, default=1.)
    p.add_argument('--seed', type=int, default=222)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--grad-clip', type=float, default=1.)
    p.add_argument('--amp', action='store_true')
    p.add_argument('--no-reliability', action='store_true')
    p.add_argument('--no-complementarity', action='store_true')
    p.add_argument('--no-temporal', action='store_true')
    p.add_argument('--no-sparse-graph', action='store_true')
    p.add_argument('--no-refinement', action='store_true')
    p.add_argument('--no-counterfactual', action='store_true', help='enhanced_v1 only')
    p.add_argument('--no-gradient-alignment', action='store_true', help='enhanced_v1 only')
    p.add_argument('--label-reweight', action='store_true', help='TRAIN labels only; optional')
    p.add_argument('--aux-weight', type=float, default=.1)
    p.add_argument('--confidence-weight', type=float, default=.05)
    p.add_argument('--factor-weight', type=float, default=.01, help='enhanced_v1 only')
    return p


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def datasets_for(kind, path):
    if not path.exists():
        raise FileNotFoundError(f'Missing {kind} pre-extracted features: {path}')
    if kind == 'mosi':
        return {s: MOSIDataset(path, split=s) for s in ('train','valid','test')}
    return load_simsv2_datasets(path)


def batch_to_device(raw, device):
    return {'text':raw['text'].to(device), 'audio':raw['audio'].to(device),
            'vision':raw['vision'].to(device), 'label':raw['label'].to(device),
            'masks': {m:raw[m+'_mask'].to(device) for m in ('text','audio','vision')}}


def make_loader(data, batch_size, shuffle, workers):
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=torch.cuda.is_available())


@torch.no_grad()
def evaluate(model, loader, device, dataset):
    model.eval()
    predictions, labels = [], []
    for raw in loader:
        batch = batch_to_device(raw, device)
        p = model(batch['text'], batch['audio'], batch['vision'], masks=batch['masks'])
        predictions.append(p.cpu().reshape(-1).numpy())
        labels.append(batch['label'].cpu().reshape(-1).numpy())
    metric = mosi_metrics if dataset == 'mosi' else simsv2_metrics
    return metric(np.concatenate(predictions), np.concatenate(labels))


def build_model(train_data, args):
    item = train_data[0]
    dims = {m+'_dim':int(item[m].shape[-1]) for m in ('text','audio','vision')}
    common = dict(d=args.hidden_dim, output_dim=1, queries=args.queries, **dims)
    variant = 'legacy' if args.legacy_baseline else args.model
    if variant == 'lite':
        cfg = dict(common, dropout=args.dropout, topk=args.topk,
                   temporal=not args.no_temporal, sparse_graph=not args.no_sparse_graph,
                   reliability=not args.no_reliability, complementarity=not args.no_complementarity,
                   refinement=not args.no_refinement, modality_dropout=args.modality_dropout,
                   routing_temperature=args.routing_temperature)
    elif variant == 'enhanced_v1':
        cfg = dict(common, dropout=args.dropout, topk=args.topk,
                   temporal=not args.no_temporal, sparse_graph=not args.no_sparse_graph,
                   reliability=not args.no_reliability, counterfactual=not args.no_counterfactual,
                   modality_dropout=args.modality_dropout)
    else:
        cfg = common
    return MODEL_TYPES[variant](**cfg), cfg, variant


def label_weight_lookup(train, kind, device):
    y = np.asarray(train.labels, dtype=np.float64).reshape(-1)
    edges = np.linspace(-3.5,3.5,8) if kind=='mosi' else np.asarray([-1.01,-.7,-.1,.1,.7,1.01])
    bins = np.clip(np.digitize(y,edges[1:-1],right=True),0,len(edges)-2)
    freq = np.bincount(bins, minlength=len(edges)-1)
    weights = 1./np.sqrt(np.maximum(freq,1))
    weights = np.clip(weights/(weights[bins].mean()+1e-12),.5,2.)
    return (torch.tensor(edges[1:-1], device=device, dtype=torch.float32),
            torch.tensor(weights, device=device, dtype=torch.float32))


def fit(dataset, args):
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError('epochs, batch size and patience must be positive')
    seed_everything(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    splits = datasets_for(dataset, args.data_path)
    loaders = {s:make_loader(splits[s],args.batch_size, s=='train',args.num_workers)
               for s in ('train','valid','test')}
    model, cfg, variant = build_model(splits['train'],args)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    total = max(args.epochs*len(loaders['train']),1)
    warmup = int(total*args.warmup_ratio)
    def lr_scale(step):
        if step < warmup:
            return max((step+1)/max(warmup,1),.001)
        t = min(1.,(step-warmup)/max(total-warmup,1))
        return .05 + .95*.5*(1+math.cos(math.pi*t))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim,lr_scale)
    amp = bool(args.amp and device.type=='cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    edges, weights = label_weight_lookup(splits['train'],dataset,device) if args.label_reweight else (None,None)
    best_mae, best_epoch, last_epoch = float('inf'), 0, 0
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({'dataset':dataset,'variant':variant,'device':str(device),
                      'sizes':{k:len(v) for k,v in splits.items()},'config':cfg,
                      'parameters':sum(p.numel() for p in model.parameters())},ensure_ascii=False),flush=True)
    for epoch in range(1,args.epochs+1):
        last_epoch = epoch
        model.train()
        running, seen = 0., 0
        for raw in loaders['train']:
            batch = batch_to_device(raw,device)
            sample_weights = (weights[torch.bucketize(batch['label'].reshape(-1),edges,right=True)]
                              if edges is not None else None)
            optim.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=amp):
                if variant == 'legacy':
                    prediction = model(batch['text'],batch['audio'],batch['vision'],masks=batch['masks'])
                    loss = task_loss(prediction,batch['label'],sample_weights=sample_weights)
                else:
                    prediction,aux = model(batch['text'],batch['audio'],batch['vision'],
                                           masks=batch['masks'],return_aux=True)
                    if variant == 'lite':
                        loss,_ = compute_loss(prediction,batch['label'],aux,sample_weights=sample_weights,
                                              aux_weight=args.aux_weight, confidence_weight=args.confidence_weight)
                    else:
                        loss,_ = old_enhanced_loss(prediction,batch['label'],aux,sample_weights=sample_weights,
                                                  aux_weight=args.aux_weight,
                                                  confidence_weight=args.confidence_weight,
                                                  factor_weight=args.factor_weight,
                                                  gradient_alignment=not args.no_gradient_alignment)
            if not torch.isfinite(loss):
                raise RuntimeError(f'nonfinite training loss at epoch {epoch}')
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip)
            scaler.step(optim)
            scaler.update()
            scheduler.step()
            n = batch['label'].shape[0]
            running += loss.detach().item()*n
            seen += n
        val = evaluate(model,loaders['valid'],device,dataset)
        print(json.dumps({'epoch':epoch,'train_loss':round(running/seen,6),
                          'valid':val,'lr':optim.param_groups[0]['lr']}),flush=True)
        if val['mae'] < best_mae - 1e-6:
            best_mae,best_epoch = val['mae'],epoch
            version = {'lite':'bchrpp_rcse_lite_v1','enhanced_v1':'bchrpp_enhanced_v1',
                       'legacy':'bchrpp_legacy'}[variant]
            torch.save({'version':version,'model_state':model.state_dict(),
                        'model_config':cfg,'epoch':epoch,'valid_metrics':val,
                        'dataset':dataset,'seed':args.seed,
                        'training_config':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}},
                       args.checkpoint)
        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}',flush=True)
            break
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    test = evaluate(model,loaders['test'],device,dataset)
    result = {'variant':variant,'seed':args.seed,'best_epoch':best_epoch,
              'last_epoch':last_epoch,'validation':ckpt['valid_metrics'],
              'test':test,'checkpoint':str(args.checkpoint),
              'note':'Test metrics must not be used for model or hyperparameter selection.'}
    print(json.dumps(result, indent=2,ensure_ascii=False),flush=True)
    args.checkpoint.with_suffix('.json').write_text(json.dumps(result, indent=2,ensure_ascii=False),encoding='utf8')
    return result


def main(dataset):
    fit(dataset,parser_for(dataset).parse_args())

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['mosi','simsv2'],required=True)
    first, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0],*remaining]
    main(first.dataset)
