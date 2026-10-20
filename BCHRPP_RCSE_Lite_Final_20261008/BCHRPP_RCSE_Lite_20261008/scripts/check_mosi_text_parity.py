"""Verify online frozen DeBERTa reproduces cached word features before training."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from bchrpp.mosi_online_text import pool_subwords
from datasets.mosi_extract import load_mosi_extract
from datasets.mosi_text_online import MOSIOnlineTextSplit


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--samples-per-split', type=int, default=3)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.samples_per_split < 1:
        raise ValueError('samples-per-split must be positive')
    from transformers import AutoModel, AutoTokenizer

    splits, meta = load_mosi_extract(args.features)
    model_path = meta['feature_models']['text']
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=True)
    if not tokenizer.is_fast:
        raise ValueError('Expected fast DeBERTa tokenizer')
    report = {'splits': meta['splits'], 'text_model': model_path, 'samples': []}
    datasets = {name: MOSIOnlineTextSplit(splits[name].samples, tokenizer)
                for name in ('train', 'valid', 'test')}
    # Check every word mapping before allocating the encoder or starting a run.
    for data in datasets.values():
        for index in range(len(data)):
            data.collate([data[index]])
    report['tokenization_checked'] = sum(len(data) for data in datasets.values())
    model = AutoModel.from_pretrained(model_path, local_files_only=True).to(args.device).eval()
    with torch.inference_mode():
        for split in ('train', 'valid', 'test'):
            data = datasets[split]
            for index in range(min(args.samples_per_split, len(data))):
                raw = data.collate([data[index]])
                model_inputs = {key: raw[f'text_{key}'].to(args.device) for key in
                                ('input_ids', 'attention_mask', 'token_type_ids')
                                if f'text_{key}' in raw}
                hidden = model(**model_inputs)
                online = pool_subwords(hidden.last_hidden_state.float(),
                                       raw['text_word_ids'].to(args.device), raw['text'].shape[1])
                error = float((online.cpu() - raw['text']).abs().max())
                report['samples'].append({'split': split, 'id': raw['id'][0],
                                          'word_count': int(raw['text'].shape[1]),
                                          'max_abs_diff': error})
                if not error <= 1e-3:
                    raise ValueError(f"{raw['id'][0]}: frozen text parity failed ({error:.6g})")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
