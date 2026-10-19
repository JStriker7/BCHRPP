"""Minimal MOSI input and regression adapter around the teacher's BCHRPP model."""

import torch.nn as nn

from bchrpp.full_model import BCHRPP
from datasets.mosi_features import FEATURE_DIMS


class MOSIRegressor(nn.Module):
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Sequential(nn.Linear(width, hidden_dim), nn.LayerNorm(hidden_dim))
            for name, width in FEATURE_DIMS.items()
        })
        self.core = BCHRPP(d=hidden_dim, classes=1)

    def forward(self, batch):
        text = self.projections["text"](batch["text"])
        audio = self.projections["audio"](batch["audio"])
        visual = self.projections["visual"](batch["visual"])
        return self.core(
            text, audio, visual,
            text_mask=batch["text_mask"],
            audio_mask=batch["audio_mask"],
            vision_mask=batch["visual_mask"],
        ).squeeze(-1)
