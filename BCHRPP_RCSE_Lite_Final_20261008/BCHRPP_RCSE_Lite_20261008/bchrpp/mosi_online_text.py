"""Word-aligned online text encoder in front of the unchanged BCHRPP."""

import torch
from torch import nn


def pool_subwords(hidden, word_ids, word_count):
    if hidden.ndim != 3 or word_ids.shape != hidden.shape[:2]:
        raise ValueError('token embeddings and word ids have incompatible shapes')
    valid = (word_ids >= 0) & (word_ids < word_count)
    index = word_ids.clamp(0, word_count - 1)
    features = torch.zeros(hidden.shape[0], word_count, hidden.shape[-1],
                           device=hidden.device, dtype=hidden.dtype)
    counts = torch.zeros(hidden.shape[0], word_count, 1,
                         device=hidden.device, dtype=hidden.dtype)
    features.scatter_add_(1, index.unsqueeze(-1).expand_as(hidden),
                          hidden * valid.unsqueeze(-1))
    counts.scatter_add_(1, index.unsqueeze(-1), valid.unsqueeze(-1).to(hidden.dtype))
    return features / counts.clamp_min(1)


class MOSIOnlineText(nn.Module):
    def __init__(self, encoder, fusion, train_text):
        super().__init__()
        self.encoder = encoder
        self.fusion = fusion
        self.train_text = train_text
        self.encoder.requires_grad_(train_text)
        if not train_text:
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        if not self.train_text:
            self.encoder.eval()
        return self

    def forward(self, text, audio, vision, masks, text_tokens, return_aux=False):
        model_inputs, word_ids = text_tokens
        if self.train_text:
            hidden = self.encoder(**model_inputs).last_hidden_state.float()
        else:
            with torch.no_grad(), torch.autocast(device_type=model_inputs['input_ids'].device.type,
                                                 enabled=False):
                hidden = self.encoder(**model_inputs).last_hidden_state.float()
        online = pool_subwords(hidden, word_ids, text.shape[1])
        return self.fusion(online, audio, vision, masks=masks, return_aux=return_aux)
