from torch import nn
from bchrpp.full_model import BCHRPP

class OnlineBCHR(nn.Module):
    def __init__(self, fusion_config):
        super().__init__()
        self.fusion = BCHRPP(**fusion_config)
        self.text_skip = nn.Sequential(nn.Dropout(0.2), nn.Linear(fusion_config['text_dim'], 1))
        nn.init.zeros_(self.text_skip[-1].weight)
        nn.init.zeros_(self.text_skip[-1].bias)

    def forward(self, text, audio, vision, masks, return_aux=False):
        fused = self.fusion(text, audio, vision, masks=masks, return_aux=return_aux)
        skip = self.text_skip(text[:, 0])
        if return_aux:
            prediction, aux = fused
            return prediction + skip, aux
        return fused + skip
