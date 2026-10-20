"""Original word-aligned MOSI data with live DeBERTa token inputs."""

import numpy as np
import torch

from datasets.mosi_extract import MOSIExtractSplit, collate_mosi_extract


def load_word_tokenizer(model_path, prefix_space=False):
    from transformers import AutoTokenizer

    options = {'use_fast': True, 'local_files_only': True}
    if prefix_space:
        options['add_prefix_space'] = True
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), **options)
    if not tokenizer.is_fast:
        raise ValueError('Online word alignment requires a fast tokenizer')
    return tokenizer


class MOSIOnlineTextSplit(MOSIExtractSplit):
    def __init__(self, samples, tokenizer):
        super().__init__(samples)
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        item = super().__getitem__(index)
        item['words'] = list(self.samples[index]['words'])
        return item

    def collate(self, batch):
        result = collate_mosi_extract(batch)
        encoded = self.tokenizer(
            [item['words'] for item in batch], is_split_into_words=True,
            padding=True, truncation=True, max_length=512, return_tensors='pt',
        )
        if not hasattr(encoded, 'word_ids'):
            raise ValueError('A fast tokenizer with word_ids() is required')
        positions = np.full(tuple(encoded['input_ids'].shape), -1, dtype=np.int64)
        for row, item in enumerate(batch):
            for token, word in enumerate(encoded.word_ids(batch_index=row)):
                if word is not None and word < len(item['words']):
                    positions[row, token] = word
            seen = set(positions[row]) - {-1}
            if seen != set(range(len(item['words']))):
                raise ValueError(f"{item['id']}: some words were truncated from online text")
        result['text_input_ids'] = encoded['input_ids']
        result['text_attention_mask'] = encoded['attention_mask']
        if 'token_type_ids' in encoded:
            result['text_token_type_ids'] = encoded['token_type_ids']
        result['text_word_ids'] = torch.from_numpy(positions)
        return result
