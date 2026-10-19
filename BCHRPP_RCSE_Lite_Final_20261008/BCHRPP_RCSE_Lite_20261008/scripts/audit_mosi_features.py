"""Validate and summarize an offline MOSI feature file before experiments.

Run on the training server with the same trusted .pt used by train_mosi_extract.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODALITIES = ("text", "audio", "visual")
SPLITS = ("train", "valid", "test")


def summarize_split(samples: list[dict]) -> dict:
    lengths = np.asarray([len(sample["text_valid"]) for sample in samples])
    labels = np.asarray([float(sample["label"]) for sample in samples])
    classes = np.clip(np.rint(labels), -3, 3).astype(int)
    alignment = Counter(str(sample.get("alignment", "unknown")) for sample in samples)
    modalities = {}
    for name in MODALITIES:
        masks = [np.asarray(sample[f"{name}_valid"], dtype=bool) for sample in samples]
        total = sum(len(mask) for mask in masks)
        valid = sum(int(mask.sum()) for mask in masks)
        modalities[name] = {
            "valid_positions": valid,
            "total_positions": total,
            "valid_fraction": valid / total if total else 0.0,
            "absent_samples": sum(not mask.any() for mask in masks),
            "partially_invalid_samples": sum(mask.any() and not mask.all() for mask in masks),
        }

    sources_missing = {
        name: sum(sample.get("sources", {}).get(name) is None for sample in samples)
        for name in ("audio", "visual")
    }
    return {
        "samples": len(samples),
        "alignment": dict(sorted(alignment.items())),
        "words": {
            "min": int(lengths.min()),
            "mean": float(lengths.mean()),
            "max": int(lengths.max()),
        },
        "label_mean": float(labels.mean()),
        "label_bins_rounded": {str(value): int((classes == value).sum()) for value in range(-3, 4)},
        "modalities": modalities,
        "missing_source_paths": sources_missing,
    }


def audit(splits: dict, meta: dict) -> dict:
    fingerprint = hashlib.sha256()
    for split in SPLITS:
        for sample in sorted(splits[split].samples, key=lambda item: item["id"]):
            fingerprint.update(
                f"{split}\t{sample['id']}\t{float(sample['label']):.9g}\n".encode("utf-8")
            )
    return {
        "format": "bhgm_sent_crc_mosi_v1",
        "feature_models": meta["feature_models"],
        "dimensions": meta["dimensions"],
        "sample_label_fingerprint": fingerprint.hexdigest(),
        "splits": {split: summarize_split(splits[split].samples) for split in SPLITS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    args = parser.parse_args()

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from datasets.mosi_extract import load_mosi_extract

    splits, meta = load_mosi_extract(args.features)
    report = audit(splits, meta)
    report["features"] = str(args.features.resolve())
    report["feature_file_bytes"] = args.features.stat().st_size
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
