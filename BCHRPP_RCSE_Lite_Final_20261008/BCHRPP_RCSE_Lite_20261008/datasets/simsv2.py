import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class SIMSv2Dataset(Dataset):
    """
    Dataset for the supervised CH-SIMSv2 pre-extracted features.

    Expected fields:
        text
        audio
        vision
        regression_labels
        audio_lengths
        vision_lengths
        id
        raw_text
    """

    def __init__(self, split_data, split):
        if split not in {"train", "valid", "test"}:
            raise ValueError(
                "split must be train, valid, or test"
            )

        self.split = split

        # Keep the original arrays in memory. They are converted to
        # float32 one sample at a time to avoid duplicating this
        # approximately 3.45 GB dataset.
        self.text = split_data["text"]
        self.audio = split_data["audio"]
        self.vision = split_data["vision"]

        self.labels = split_data[
            "regression_labels"
        ]

        self.text_labels = split_data.get(
            "regression_labels_T"
        )
        self.audio_labels = split_data.get(
            "regression_labels_A"
        )
        self.vision_labels = split_data.get(
            "regression_labels_V"
        )

        self.audio_lengths = split_data[
            "audio_lengths"
        ]
        self.vision_lengths = split_data[
            "vision_lengths"
        ]

        self.ids = split_data["id"]
        self.raw_text = split_data["raw_text"]

    def __len__(self):
        return len(self.labels)

    @staticmethod
    def _to_tensor(values):
        values = np.asarray(
            values,
            dtype=np.float32,
        )

        values = np.nan_to_num(
            values,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        return torch.from_numpy(values)

    def __getitem__(self, index):
        text = self._to_tensor(
            self.text[index]
        )
        audio = self._to_tensor(
            self.audio[index]
        )
        vision = self._to_tensor(
            self.vision[index]
        )

        label = torch.tensor(
            [float(self.labels[index])],
            dtype=torch.float32,
        )

        # Padded timesteps are all zeros.
        text_mask = (
            text.abs()
            .sum(dim=-1)
            .ne(0)
        )

        audio_mask = (
            audio.abs()
            .sum(dim=-1)
            .ne(0)
        )

        vision_mask = (
            vision.abs()
            .sum(dim=-1)
            .ne(0)
        )

        sample = {
            "text": text,
            "audio": audio,
            "vision": vision,
            "label": label,
            "text_mask": text_mask,
            "audio_mask": audio_mask,
            "vision_mask": vision_mask,
            "audio_length": int(
                self.audio_lengths[index]
            ),
            "vision_length": int(
                self.vision_lengths[index]
            ),
            "id": str(self.ids[index]),
            "raw_text": str(
                self.raw_text[index]
            ),
        }

        if self.text_labels is not None:
            sample["text_label"] = torch.tensor(
                [float(self.text_labels[index])],
                dtype=torch.float32,
            )

        if self.audio_labels is not None:
            sample["audio_label"] = torch.tensor(
                [float(self.audio_labels[index])],
                dtype=torch.float32,
            )

        if self.vision_labels is not None:
            sample["vision_label"] = torch.tensor(
                [float(self.vision_labels[index])],
                dtype=torch.float32,
            )

        return sample


def load_simsv2_datasets(path):
    """
    Load the PKL only once and construct the three official splits.

    Loading the file separately for train/valid/test would consume
    excessive memory because the complete PKL is about 3.45 GB.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("rb") as file:
        data = pickle.load(file)

    required_splits = {
        "train",
        "valid",
        "test",
    }

    missing = required_splits - data.keys()

    if missing:
        raise ValueError(
            "missing data splits: "
            + ", ".join(sorted(missing))
        )

    return {
        split: SIMSv2Dataset(
            data[split],
            split=split,
        )
        for split in (
            "train",
            "valid",
            "test",
        )
    }