"""Adapter from ``extract/extract_mosi_features.py`` output to RCSE-Lite input.

The offline MOSI extractor writes a single ``.pt`` file whose ``train`` /
``valid`` / ``test`` values are lists of per-sample dicts.  Each sample carries
word-aligned features and per-modality validity masks:

    text    -> DeBERTa-v3-large, [T, 1024]
    audio   -> LAION-CLAP HTSAT, [T, 512]
    visual  -> CLIP ViT-L/14@336 projection, [T, 768]

This module validates that exact contract and converts it to the padded
``[B, T, D]`` tensors plus ``masks`` dict consumed by
``bchrpp.full_model.BCHRPP``.  The model names the visual modality ``vision``
while the extractor names it ``visual``, so the key is remapped here.

Do NOT mix in ``en_extract.py`` / ``extract_mosi_reference_av.py`` features:
their dimensions and data conventions are different.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

FEATURE_DIMS = {"text": 1024, "audio": 512, "vision": 768}
SPLITS = ("train", "valid", "test")
# extractor key -> model key for the same modality
EXTRACT_KEY = {"text": "text", "audio": "audio", "vision": "visual"}


class MOSIExtractSplit(Dataset):
    """One split of the extract feature file, converted to tensors on access."""

    def __init__(self, samples: List[dict]):
        self.samples = samples
        self.labels = np.asarray(
            [float(sample["label"]) for sample in samples], dtype=np.float64
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        item = {
            "id": sample["id"],
            "label": torch.tensor(float(sample["label"]), dtype=torch.float32),
        }
        for model_name in FEATURE_DIMS:
            extract_name = EXTRACT_KEY[model_name]
            values = np.asarray(sample[extract_name], dtype=np.float32)
            mask = np.asarray(
                sample[f"{extract_name}_valid"], dtype=np.float32
            ).astype(bool)
            item[model_name] = torch.from_numpy(values)
            item[f"{model_name}_mask"] = torch.from_numpy(mask)
        return item


def collate_mosi_extract(batch: List[Dict[str, torch.Tensor]]) -> Dict:
    """Pad a variable-length batch to ``[B, T, D]`` per modality."""
    result = {
        "id": [sample["id"] for sample in batch],
        "label": torch.stack([sample["label"] for sample in batch]).reshape(-1, 1),
    }
    for name in FEATURE_DIMS:
        result[name] = pad_sequence(
            [sample[name] for sample in batch], batch_first=True
        )
        # Padded mask positions are False, i.e. invalid.
        result[f"{name}_mask"] = pad_sequence(
            [sample[f"{name}_mask"] for sample in batch], batch_first=True
        )
    return result


def load_mosi_extract(path) -> tuple[Dict[str, MOSIExtractSplit], dict]:
    """Validate and load the extract feature file.

    Returns ``(splits, meta)`` where ``splits`` maps ``train/valid/test`` to
    datasets and ``meta`` summarises the file for logging.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"MOSI extract feature file not found: {path}")
    # The extractor stores NumPy arrays, so weights_only=False is required.
    # Only open feature files produced by a trusted copy of the extractor.
    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, dict):
        raise ValueError("MOSI extract features must be a dict of splits")
    metadata = data.get("_metadata", {})
    if metadata.get("format") != "bhgm_sent_crc_mosi_v1":
        raise ValueError(
            "Expected extract_mosi_features.py offline format "
            f"'bhgm_sent_crc_mosi_v1'; got {metadata.get('format')!r}. "
            "Do not mix in en_extract.py or reference-av features."
        )
    if metadata.get("dimensions") != {"text": 1024, "audio": 512, "visual": 768}:
        raise ValueError(
            f"Unexpected feature dimensions {metadata.get('dimensions')}; "
            f"expected {FEATURE_DIMS}."
        )

    seen_ids = set()
    splits: Dict[str, MOSIExtractSplit] = {}
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

            label = float(np.asarray(sample.get("label")))
            if not np.isfinite(label):
                raise ValueError(f"{uid}: label must be finite")
            if not -3.0 <= label <= 3.0:
                print(f"[warning] {uid}: label {label} outside [-3, 3]")

            for model_name, width in FEATURE_DIMS.items():
                extract_name = EXTRACT_KEY[model_name]
                values = np.asarray(sample.get(extract_name))
                mask = np.asarray(sample.get(f"{extract_name}_valid"))
                if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != width:
                    raise ValueError(
                        f"{uid}: {extract_name} must be nonempty [T,{width}]; "
                        f"got {values.shape}"
                    )
                if mask.shape != (values.shape[0],) or not np.isin(mask, (0, 1)).all():
                    raise ValueError(f"{uid}: invalid {extract_name}_valid mask")
                if not np.isfinite(values).all():
                    raise ValueError(f"{uid}: non-finite {extract_name} features")
                if model_name == "text" and not mask.any():
                    raise ValueError(f"{uid}: text modality has no valid positions")

            words = sample.get("words")
            if isinstance(words, (list, tuple)) and words:
                text_len = np.asarray(sample[EXTRACT_KEY["text"]]).shape[0]
                if len(words) != text_len:
                    raise ValueError(
                        f"{uid}: word/text length disagree ({len(words)} vs {text_len})"
                    )

        splits[split] = MOSIExtractSplit(samples)

    meta = {
        "splits": {name: len(splits[name]) for name in SPLITS},
        "feature_models": metadata.get("models", {}),
        "dimensions": metadata.get("dimensions"),
    }
    return splits, meta
