
import torch
from torch.utils.data import Dataset

class MultimodalDataset(Dataset):
    """Unified loader base for MOSI/MOSEI/SIMS/MELD/URFUNNY/IEMOCAP."""
    def __init__(self,path):
        self.data=torch.load(path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,i):
        x=self.data[i]
        return {
            "text":x["text"],
            "audio":x["audio"],
            "vision":x["vision"],
            "label":x["label"]
        }
