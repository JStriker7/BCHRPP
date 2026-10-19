"""Paired-seed reference / ablation runner; no test-guided hyperparameter search."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

MODES = {
    'legacy':['--model','legacy'],
    'enhanced_v1':['--model','enhanced_v1'],
    'lite':[],
    'no_reliability':['--no-reliability'],
    'no_complementarity':['--no-complementarity'],
    'no_temporal':['--no-temporal'],
    'dense_graph':['--no-sparse-graph'],
    'no_refinement':['--no-refinement'],
    'no_modality_dropout':['--modality-dropout','0'],
    'no_quality_aux':['--aux-weight','0','--confidence-weight','0'],
}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',choices=['mosi','simsv2'],required=True)
    p.add_argument('--data-path',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=60)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--hidden-dim',type=int,default=128)
    p.add_argument('--queries',type=int,default=4)
    p.add_argument('--seeds',default='222,128,2026')
    p.add_argument('--modes',default='lite,no_reliability,no_complementarity,dense_graph,no_refinement')
    p.add_argument('--amp',action='store_true')
    args=p.parse_args()
    modes=[x.strip() for x in args.modes.split(',')]
    seeds=[int(x.strip()) for x in args.seeds.split(',')]
    for mode in modes:
        if mode not in MODES:raise ValueError('unknown ablation '+mode)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    results=[]
    for mode in modes:
        for seed in seeds:
            checkpoint=args.output_dir/f'{args.dataset}_{mode}_{seed}.pt'
            run=[sys.executable,str(Path(__file__).parent/('train_mosi.py' if args.dataset=='mosi' else 'train_simsv2.py')),
                 '--data-path',str(args.data_path),'--checkpoint',str(checkpoint),
                 '--epochs',str(args.epochs),'--batch-size',str(args.batch_size),
                 '--hidden-dim',str(args.hidden_dim),'--queries',str(args.queries),
                 '--seed',str(seed),*MODES[mode]]
            if args.amp:run+=['--amp']
            print('RUN',mode,'seed',seed,flush=True)
            subprocess.run(run,check=True)
            summary=json.loads(checkpoint.with_suffix('.json').read_text(encoding='utf8'))
            results.append({'mode':mode,'seed':seed,'best_epoch':summary['best_epoch'],
                            **{'val_'+k:v for k,v in summary['validation'].items()},
                            **{'test_'+k:v for k,v in summary['test'].items()}})
            with (args.output_dir/'results.csv').open('w',newline='',encoding='utf8') as file:
                writer=csv.DictWriter(file,fieldnames=results[0].keys())
                writer.writeheader();writer.writerows(results)
    print('completed',len(results),'runs; raw results in',args.output_dir)

if __name__=='__main__':main()
