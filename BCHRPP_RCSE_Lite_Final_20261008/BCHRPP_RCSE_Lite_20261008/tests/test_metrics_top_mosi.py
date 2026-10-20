"""Check the training evaluator calls the teacher's MOSI regression protocol."""

import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from metrics.metricsTop import MetricsTop
from scripts.run_mosi_extract_ablations import METRIC_KEYS
from scripts.train_mosi_extract import METRIC_PROTOCOL, VERSION, evaluate, recover_trusted_checkpoint


class _TextPredictionModel:
    def eval(self):
        return self

    def __call__(self, text, audio, vision, masks=None, text_tokens=None):
        return text[:, 0, :1]


class _RecoverableModel(torch.nn.Module):
    def forward(self, text, audio, vision, masks=None, text_tokens=None):
        return text[:, 0, :1]


class MetricsTopMosiTest(unittest.TestCase):
    def test_evaluate_matches_teacher_and_keeps_unrounded_selection_mae(self):
        predictions = torch.tensor([-3.6, 0.0, 0.2, -0.2, 0.51, 2.8])
        labels = torch.tensor([-3.0, -1.0, 0.0, 1.0, 1.0, 3.0])
        raw = {
            "text": predictions.reshape(-1, 1, 1),
            "audio": torch.zeros(6, 1, 1),
            "vision": torch.zeros(6, 1, 1),
            "label": labels.reshape(-1, 1),
            "text_mask": torch.ones(6, 1, dtype=torch.bool),
            "audio_mask": torch.ones(6, 1, dtype=torch.bool),
            "vision_mask": torch.ones(6, 1, dtype=torch.bool),
            "text_input_ids": torch.ones(6, 1, dtype=torch.long),
            "text_attention_mask": torch.ones(6, 1, dtype=torch.long),
            "text_word_ids": torch.zeros(6, 1, dtype=torch.long),
        }

        report, selection_mae = evaluate(_TextPredictionModel(), [raw], torch.device("cpu"))
        teacher = MetricsTop("regression").getMetics("MOSI")(predictions, labels)
        self.assertEqual(report, teacher)
        self.assertTrue(all(type(value) is float for value in report.values()))
        saved = io.BytesIO()
        torch.save({"valid_metrics": report}, saved)
        saved.seek(0)
        self.assertEqual(torch.load(saved, weights_only=True)["valid_metrics"], report)
        self.assertEqual(set(METRIC_KEYS), set(report))
        self.assertEqual(report["Mult_acc_7"], 0.6667)
        self.assertEqual(report["Has0_acc_2"], 0.6667)
        self.assertEqual(report["Non0_acc_2"], 0.8)
        self.assertEqual(report["MAE"], 0.615)
        expected = np.mean(np.abs(
            predictions.numpy().astype(np.float64) - labels.numpy().astype(np.float64)
        ))
        self.assertEqual(selection_mae, expected)

    def test_recovery_keeps_original_and_produces_safe_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.pt"
            features.touch()
            checkpoint = root / "mosi_lite_seed222.pt"
            predictions = torch.tensor([0.2, -0.4, 1.2])
            labels = torch.tensor([0.0, -1.0, 1.0])
            batch = {
                "text": predictions.reshape(-1, 1, 1),
                "audio": torch.zeros(3, 1, 1), "vision": torch.zeros(3, 1, 1),
                "label": labels.reshape(-1, 1),
                "text_mask": torch.ones(3, 1, dtype=torch.bool),
                "audio_mask": torch.ones(3, 1, dtype=torch.bool),
                "vision_mask": torch.ones(3, 1, dtype=torch.bool),
                "text_input_ids": torch.ones(3, 1, dtype=torch.long),
                "text_attention_mask": torch.ones(3, 1, dtype=torch.long),
                "text_word_ids": torch.zeros(3, 1, dtype=torch.long),
            }
            metrics, selection = evaluate(_RecoverableModel(), [batch], torch.device("cpu"))
            old_metrics = dict(metrics)
            old_metrics["Corr"] = np.float64(old_metrics["Corr"])
            torch.save({
                "version": VERSION, "model_state": {},
                "model_config": {"text_mode": "frozen", "text_model": "fake"},
                "epoch": 8, "valid_metrics": old_metrics,
                "metric_protocol": METRIC_PROTOCOL, "selection_val_mae": selection,
                "dataset": "MOSI", "seed": 222, "features": str(features.resolve()),
            }, checkpoint)
            original_bytes = checkpoint.read_bytes()
            args = Namespace(checkpoint=checkpoint, features=features, seed=222,
                             epochs=60, patience=15, batch_size=24, num_workers=0,
                             text_mode="frozen", text_model="fake", modalities="full")
            with patch("scripts.train_mosi_extract.model_from_config",
                       return_value=_RecoverableModel()), \
                    patch("scripts.train_mosi_extract.make_loader", return_value=[batch]):
                result = recover_trusted_checkpoint(
                    args, {"valid": object(), "test": object()}, torch.device("cpu")
                )
            self.assertEqual(checkpoint.read_bytes(), original_bytes)
            safe = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
            self.assertTrue(all(type(value) is float for value in safe["valid_metrics"].values()))
            self.assertEqual(result["last_epoch"], 23)
            sidecar = json.loads(checkpoint.with_suffix(".json").read_text(encoding="utf8"))
            self.assertEqual(sidecar["checkpoint"], result["checkpoint"])


if __name__ == "__main__":
    unittest.main()
