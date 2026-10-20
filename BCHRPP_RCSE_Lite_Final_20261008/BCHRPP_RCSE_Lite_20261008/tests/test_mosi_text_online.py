"""Online word pooling and frozen-gradient checks without pretrained weights."""

import unittest
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from bchrpp.mosi_online_text import MOSIOnlineText, pool_subwords
from datasets.mosi_text_online import MOSIOnlineTextSplit


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(6, 2)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeFusion(nn.Module):
    def forward(self, text, audio, vision, masks, return_aux=False):
        return text


class FakeTokens(dict):
    def word_ids(self, batch_index):
        return [-1, 0, 0, 1, -1]


class FakeTokenizer:
    def __call__(self, words, **kwargs):
        assert kwargs['is_split_into_words'] and kwargs['max_length'] == 512
        return FakeTokens(input_ids=torch.zeros(len(words), 5, dtype=torch.long),
                          attention_mask=torch.ones(len(words), 5, dtype=torch.long))


class OnlineTextTest(unittest.TestCase):
    def test_online_collate_retains_word_level_alignment_and_modalities(self):
        sample = {'id': 'sample_1', 'label': 1., 'words': ['hello', 'world']}
        for name, width in (('text', 1024), ('audio', 512), ('visual', 768)):
            sample[name] = np.zeros((2, width), np.float32)
            sample[f'{name}_valid'] = np.ones(2, np.float32)
        data = MOSIOnlineTextSplit([sample], FakeTokenizer())
        result = data.collate([data[0]])
        self.assertEqual(tuple(result['text_word_ids'].shape), (1, 5))
        self.assertEqual(result['text_word_ids'].tolist(), [[-1, 0, 0, 1, -1]])
        self.assertEqual(tuple(result['audio'].shape), (1, 2, 512))
        self.assertEqual(tuple(result['vision'].shape), (1, 2, 768))

    def test_subword_mean_and_padding(self):
        hidden = torch.tensor([[[99., 99.], [2., 4.], [4., 6.], [8., 10.], [99., 99.]]])
        result = pool_subwords(hidden, torch.tensor([[-1, 0, 0, 1, -1]]), 3)
        torch.testing.assert_close(result[0], torch.tensor([[3., 5.], [8., 10.], [0., 0.]]))

    def test_only_unfrozen_encoder_receives_gradients(self):
        ids = torch.tensor([[1, 2, 3]])
        text = torch.zeros(1, 2, 2)
        words = torch.tensor([[-1, 0, 1]])
        for train_text in (False, True):
            model = MOSIOnlineText(FakeEncoder(), FakeFusion(), train_text)
            model.train()
            self.assertEqual(model.encoder.training, train_text)
            value = model(text, text, text, {},
                          ({'input_ids': ids, 'attention_mask': torch.ones_like(ids)}, words))
            self.assertEqual(tuple(value.shape), (1, 2, 2))
            if train_text:
                value.sum().backward()
                self.assertIsNotNone(model.encoder.embedding.weight.grad)
            else:
                self.assertFalse(value.requires_grad)
                self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))


if __name__ == '__main__':
    unittest.main()
