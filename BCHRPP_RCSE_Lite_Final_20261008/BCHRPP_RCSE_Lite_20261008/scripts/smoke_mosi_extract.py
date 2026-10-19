"""Synthetic end-to-end smoke test for the extract -> RCSE-Lite bridge.

Generates a tiny extract-format ``.pt`` in a temp directory, then trains
RCSE-Lite for a couple of CPU epochs and prints the metric dict.  No real MOSI
features or GPU are required, so it can confirm the new adapter works before
running the full extraction + training.

Run from the project root:

    python scripts/smoke_mosi_extract.py
"""
import argparse
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from datasets.mosi_extract import FEATURE_DIMS, SPLITS
from scripts.train_mosi_extract import fit, load_mosi_extract, parser

EXTRACT_KEY = {"text": "text", "audio": "audio", "vision": "visual"}


def make_features(path: Path) -> None:
    rng = np.random.default_rng(0)
    data = {}
    counts = {"train": 8, "valid": 4, "test": 4}
    for split in SPLITS:
        samples = []
        for i in range(counts[split]):
            length = int(rng.integers(3, 8))
            label = float(rng.uniform(-3, 3))
            sample = {"id": f"{split}_{i}", "words": ["w"] * length,
                      "label": np.float32(label), "alignment": "uniform"}
            for model_name, width in FEATURE_DIMS.items():
                extract_name = EXTRACT_KEY[model_name]
                sample[extract_name] = rng.normal(size=(length, width)).astype(np.float32)
                sample[f"{extract_name}_valid"] = np.ones(length, dtype=np.float32)
            samples.append(sample)
        data[split] = samples
    data["_metadata"] = {
        "format": "bhgm_sent_crc_mosi_v1",
        "dimensions": {"text": 1024, "audio": 512, "visual": 768},
        "models": {"text": "synthetic", "audio": "synthetic", "visual": "synthetic"},
        "splits": counts,
    }
    torch.save(data, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--hidden-dim", type=int, default=32)
    ap.add_argument("--queries", type=int, default=2)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as td:
        features = Path(td) / "features.pt"
        checkpoint = Path(td) / "best.pt"
        make_features(features)
        splits, meta = load_mosi_extract(features)
        print("loaded:", meta)

        train_args = parser().parse_args([
            "--features", str(features),
            "--checkpoint", str(checkpoint),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--hidden-dim", str(args.hidden_dim),
            "--queries", str(args.queries),
            "--topk", str(args.topk),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--patience", "5",
        ])
        result = fit(train_args, splits, torch.device("cpu"))
        print("SMOKE OK; test=", result["test"])


if __name__ == "__main__":
    main()
