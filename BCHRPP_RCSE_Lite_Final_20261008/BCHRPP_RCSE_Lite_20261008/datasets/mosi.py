import pickle

import numpy as np
import torch
from torch.utils.data import Dataset


class MOSIDataset(Dataset):
    def __init__(self, path, split="train"):
        if split not in {"train", "valid", "test"}:
            raise ValueError("split must be train, valid, or test")

        with open(path, "rb") as file:
            data = pickle.load(file)

        part = data[split]

        self.text = self._clean(part["text"])
        self.audio = self._clean(part["audio"])
        self.vision = self._clean(part["vision"])
        self.labels = self._clean(part["regression_labels"]).reshape(-1, 1)

        self.audio_lengths = np.asarray(part["audio_lengths"], dtype=np.int64)
        self.vision_lengths = np.asarray(part["vision_lengths"], dtype=np.int64)
        self.ids = part["id"]
        self.raw_text = part["raw_text"]

    @staticmethod
    def _clean(values):
        values = np.asarray(values, dtype=np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        text = torch.from_numpy(self.text[index])
        audio = torch.from_numpy(self.audio[index])
        vision = torch.from_numpy(self.vision[index])

        text_mask = text.abs().sum(dim=-1).ne(0)
        audio_mask = audio.abs().sum(dim=-1).ne(0)
        vision_mask = vision.abs().sum(dim=-1).ne(0)

        return {
            "text": text,
            "audio": audio,
            "vision": vision,
            "label": torch.from_numpy(self.labels[index]),
            "text_mask": text_mask,
            "audio_mask": audio_mask,
            "vision_mask": vision_mask,
            "id": self.ids[index],
            "raw_text": self.raw_text[index],
        }