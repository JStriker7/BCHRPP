"""Adapter for the unmodified extract/extract_mosi_features.py output."""

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


FEATURE_DIMS = {"text": 1024, "audio": 512, "visual": 768}
SPLITS = ("train", "valid", "test")


class MOSIFeatureSplit(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        result = {"id": sample["id"], "label": torch.tensor(sample["label"], dtype=torch.float32)}
        for modality in FEATURE_DIMS:
            result[modality] = torch.as_tensor(sample[modality], dtype=torch.float32)
            result[f"{modality}_mask"] = torch.as_tensor(
                sample[f"{modality}_valid"], dtype=torch.bool
            )
        return result


def collate_mosi(batch):
    result = {
        "id": [sample["id"] for sample in batch],
        "label": torch.stack([sample["label"] for sample in batch]),
    }
    for modality in FEATURE_DIMS:
        result[modality] = pad_sequence(
            [sample[modality] for sample in batch], batch_first=True
        )
        result[f"{modality}_mask"] = pad_sequence(
            [sample[f"{modality}_mask"] for sample in batch], batch_first=True
        )
    return result


def load_mosi_features(path):
    """Validate the exact offline feature contract before allocating GPU memory."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MOSI feature file not found: {path}")
    # The extractor stores NumPy arrays, so weights_only=False is required.
    # Only open feature files produced by a trusted copy of the extractor.
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError("MOSI features must be a dictionary of splits")
    metadata = data.get("_metadata", {})
    if metadata.get("format") != "bhgm_sent_crc_mosi_v1":
        raise ValueError("Expected extract_mosi_features.py offline feature format")
    if metadata.get("dimensions") != FEATURE_DIMS:
        raise ValueError(
            f"Expected offline feature dimensions {FEATURE_DIMS}, got {metadata.get('dimensions')}"
        )

    seen_ids = set()
    alignments = Counter()
    splits = {}
    for split in SPLITS:
        samples = data.get(split)
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"MOSI split {split!r} is missing or empty")
        if metadata.get("splits", {}).get(split) != len(samples):
            raise ValueError(f"MOSI split {split!r} count disagrees with metadata")
        for sample in samples:
            uid = sample.get("id")
            if not isinstance(uid, str) or not uid or uid in seen_ids:
                raise ValueError(f"Missing or duplicate MOSI sample id: {uid!r}")
            seen_ids.add(uid)
            label = np.asarray(sample.get("label"))
            if label.ndim != 0 or not np.isfinite(label) or not -3 <= float(label) <= 3:
                raise ValueError(f"{uid}: label must be a finite score in [-3, 3]")
            for modality, width in FEATURE_DIMS.items():
                values = np.asarray(sample.get(modality))
                mask = np.asarray(sample.get(f"{modality}_valid"))
                if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != width:
                    raise ValueError(f"{uid}: {modality} must be nonempty [T,{width}]")
                if mask.shape != (values.shape[0],) or not np.isin(mask, (0, 1)).all():
                    raise ValueError(f"{uid}: invalid {modality}_valid mask")
                if not mask.any() or not np.isfinite(values).all():
                    raise ValueError(f"{uid}: missing or non-finite {modality} features")
            if len(sample.get("words", [])) != len(sample["text"]):
                raise ValueError(f"{uid}: word and text feature lengths disagree")
            alignments[sample.get("alignment", "unknown")] += 1
        splits[split] = MOSIFeatureSplit(samples)
    return splits, {"splits": {name: len(splits[name]) for name in SPLITS},
                    "alignment": dict(alignments), "feature_models": metadata.get("models", {})}
