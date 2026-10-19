"""Evaluate historical, enhanced_v1 and RCSE-Lite MOSI checkpoints safely."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import torch
from torch.utils.data import DataLoader
from datasets.mosi import MOSIDataset
from scripts.train_regression import MODEL_TYPES, VERSION_TYPES, evaluate

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data-path',type=Path,default=ROOT/'data'/'MOSI'/'Processed'/'unaligned_50.pkl')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints'/'mosi_best.pt')
    p.add_argument('--batch-size',type=int,default=32)
    args=p.parse_args()
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    version=ckpt.get('version', 'legacy')
    # Historical checkpoints may omit a version; only treat missing as legacy.
    if version not in VERSION_TYPES and version != 'legacy':
        raise ValueError(f'unknown checkpoint version {version!r}; refusing incompatible weights')
    variant=VERSION_TYPES.get(version,'legacy')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=MODEL_TYPES[variant](**ckpt['model_config']).to(device)
    model.load_state_dict(ckpt['model_state'],strict=True)
    score=evaluate(model,DataLoader(MOSIDataset(args.data_path,'test'),batch_size=args.batch_size),device,'mosi')
    print(f'checkpoint={args.checkpoint}, version={version}, epoch={ckpt.get("epoch","unknown")}')
    for key,val in score.items(): print(f'{key}: {val:.6f}')

if __name__ == '__main__':main()
