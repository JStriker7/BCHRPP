"""CPU smoke tests; run after installing the server training environment."""

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

from bchrpp.mosi_model import MOSIRegressor
from datasets.mosi_features import FEATURE_DIMS, collate_mosi, load_mosi_features
from metrics.evaluation import mosi_metrics


ROOT = Path(__file__).resolve().parents[1]


def sample(uid, length, label):
    rng = np.random.default_rng(len(uid) + length)
    row = {"id": uid, "words": ["word"] * length, "label": np.float32(label),
           "alignment": "uniform"}
    for modality, width in FEATURE_DIMS.items():
        row[modality] = rng.normal(size=(length, width)).astype(np.float32)
        row[f"{modality}_valid"] = np.ones(length, dtype=np.float32)
    return row


def save_features(path, train=None):
    splits = {"train": train or [sample("train_1", 2, -1), sample("train_2", 3, 1)],
              "valid": [sample("valid_1", 2, 0.5)],
              "test": [sample("test_1", 1, -0.5)]}
    splits["_metadata"] = {"format": "bhgm_sent_crc_mosi_v1",
                           "dimensions": FEATURE_DIMS.copy(),
                           "models": {"text": "synthetic", "audio": "synthetic",
                                      "visual": "synthetic"},
                           "splits": {key: len(splits[key]) for key in ("train", "valid", "test")}}
    torch.save(splits, path)


class MOSIPipelineTests(unittest.TestCase):
    def test_metrics_keep_raw_mae(self):
        y = [-3, -1, 0, 1, 3]
        p = [-4, -0.7, 0.2, 1.4, 4]
        result = mosi_metrics(y, p)
        self.assertEqual(result["Acc-7"], 1.0)
        self.assertAlmostEqual(result["MAE"], 0.58)
        self.assertEqual(result["Acc-2_nonzero"], 1.0)

    def test_loader_masks_and_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.pt"
            save_features(path)
            splits, report = load_mosi_features(path)
            batch = collate_mosi([splits["train"][0], splits["train"][1]])
            self.assertEqual(batch["text"].shape, (2, 3, 1024))
            self.assertFalse(batch["text_mask"][0, 2])
            self.assertEqual(report["splits"], {"train": 2, "valid": 1, "test": 1})
            model = MOSIRegressor(hidden_dim=16).eval()
            with torch.no_grad():
                padded = model(batch)[0]
                unpadded = model(collate_mosi([splits["train"][0]]))[0]
            self.assertTrue(torch.allclose(padded, unpadded, atol=1e-5))

    def test_reject_missing_modality(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "features.pt"
            row = sample("train_1", 2, -1)
            row["audio_valid"][:] = 0
            save_features(path, train=[row])
            with self.assertRaisesRegex(ValueError, "missing or non-finite audio"):
                load_mosi_features(path)

    def test_one_epoch_train_and_evaluate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features, config, run = root / "features.pt", root / "config.yaml", root / "run"
            save_features(features)
            config.write_text(yaml.safe_dump({
                "hidden_dim": 16, "batch_size": 2, "epochs": 1, "lr": 1e-4,
                "amp": False, "seed": 3, "weight_decay": 0.01,
                "grad_clip": 1.0, "huber_beta": 1.0, "patience": 1,
            }), encoding="utf-8")
            command = [sys.executable, "-m", "scripts.train", "--features", str(features),
                       "--config", str(config), "--device", "cpu", "--run-dir", str(run)]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(result["best_epoch"], 1)
            self.assertIn("Acc-7", result["test"])
            subprocess.run([sys.executable, "-m", "scripts.train", "--features", str(features),
                            "--evaluate", str(run / "best.pt"), "--device", "cpu"],
                           cwd=ROOT, check=True, capture_output=True, text=True)

    def test_early_stopping_respects_patience(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features, config, run = root / "features.pt", root / "config.yaml", root / "run"
            save_features(features)
            config.write_text(yaml.safe_dump({
                "hidden_dim": 16, "batch_size": 2, "epochs": 8, "lr": 1e-30,
                "amp": False, "seed": 3, "weight_decay": 0.01,
                "grad_clip": 1.0, "huber_beta": 1.0, "patience": 2,
            }), encoding="utf-8")
            command = [sys.executable, "-m", "scripts.train", "--features", str(features),
                       "--config", str(config), "--device", "cpu", "--run-dir", str(run)]
            completed = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            epochs = [int(line.split(" ", 1)[0][len("epoch="):])
                      for line in completed.stdout.splitlines() if line.startswith("epoch=")]
            result = json.loads((run / "result.json").read_text(encoding="utf-8"))
            self.assertEqual(epochs, [1, 2, 3])
            self.assertEqual(result["best_epoch"], 1)
            self.assertEqual(result["config"]["epochs"], 8)
            self.assertEqual(result["config"]["patience"], 2)
            self.assertTrue((run / "best.pt").is_file())


if __name__ == "__main__":
    unittest.main()
