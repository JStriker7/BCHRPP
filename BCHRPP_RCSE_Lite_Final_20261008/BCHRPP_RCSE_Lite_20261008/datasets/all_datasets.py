"""Historical placeholder dataset class names, NOT implemented data pipelines.

Only datasets.mosi.MOSIDataset and datasets.simsv2.SIMSv2Dataset are verified
for pre-extracted regression in this project.
"""

from .base import MultimodalDataset

class MOSI(MultimodalDataset): pass
class MOSEI(MultimodalDataset): pass
class CHSIMS(MultimodalDataset): pass
class CHSIMSv2(MultimodalDataset): pass
class MELD(MultimodalDataset): pass
class URFUNNY(MultimodalDataset): pass
class IEMOCAP(MultimodalDataset): pass
