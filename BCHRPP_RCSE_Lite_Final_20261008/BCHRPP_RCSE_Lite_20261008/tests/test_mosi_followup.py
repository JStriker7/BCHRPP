"""Fixed-grid provenance, strict reuse, masks, and optimizer tests."""

import contextlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from bchrpp.full_model import BCHRPP
from bchrpp.mosi_online_text import MOSIOnlineText
from datasets.mosi_text_online import load_word_tokenizer
from losses.bchrpp_loss import compute_loss
from scripts.mosi_followup import (METRIC_KEYS, SEEDS, TRIALS, check_original_modalities,
                                   make_manifest, run_trial, sample_label_fingerprint,
                                   train_command, validate_inputs, variant_config)
from scripts.train_mosi_extract import batch_to_device, make_optimizer


def sample(uid, label=0.):
    return {"id": uid, "label": label, "words": ["hello", "world"],
            "text": np.zeros((2, 1024), np.float32),
            "audio": np.ones((2, 512), np.float32),
            "visual": np.ones((2, 768), np.float32),
            "audio_valid": np.ones(2, np.float32),
            "visual_valid": np.ones(2, np.float32)}


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10, 4)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class MosiFollowupTest(unittest.TestCase):
    def test_variants_are_fixed_and_learning_rates_are_explicit(self):
        self.assertEqual([v.name for v in TRIALS["modality"]],
                         ["full", "no_audio", "no_vision", "text_only"])
        self.assertEqual([v.text_lr for v in TRIALS["text_lr"]],
                         [1e-5, 2e-5, 3e-5])
        self.assertEqual([v.text_mode for v in TRIALS["roberta"]],
                         ["frozen", "unfrozen"])
        self.assertTrue(all(v.tokenizer_prefix_space for v in TRIALS["roberta"]))
        manifest = {"features": "/tmp/features.pt", "features_sha256": "a" * 64,
                    "text_model": "/tmp/model"}
        for variant in TRIALS["text_lr"]:
            manifest["trial"] = variant.name
            command = train_command(manifest, variant, 222, Path("run.pt"))
            self.assertEqual(float(command[command.index("--text-lr") + 1]),
                             variant.text_lr)
            self.assertEqual(float(command[command.index("--lr") + 1]), 3e-4)

    def test_local_model_prefix_space_is_explicit(self):
        tokenizer = SimpleNamespace(is_fast=True)
        with patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer) as load:
            self.assertIs(load_word_tokenizer("roberta-large", True), tokenizer)
            self.assertTrue(load.call_args.kwargs["add_prefix_space"])
            self.assertTrue(load.call_args.kwargs["local_files_only"])
            load_word_tokenizer("deberta-v3-large")
            self.assertNotIn("add_prefix_space", load.call_args.kwargs)

    def test_fingerprint_is_order_independent_and_original_av_are_exact(self):
        first = {split: SimpleNamespace(samples=[sample(split + "b", 1.),
                                                 sample(split + "a", -1.)])
                 for split in ("train", "valid", "test")}
        second = {split: SimpleNamespace(samples=list(reversed(data.samples)))
                  for split, data in first.items()}
        self.assertEqual(sample_label_fingerprint(first), sample_label_fingerprint(second))
        check_original_modalities(first, second)
        second["valid"].samples[0] = dict(second["valid"].samples[0])
        second["valid"].samples[0]["visual_valid"] = np.zeros(2, np.float32)
        with self.assertRaisesRegex(ValueError, "visual_valid"):
            check_original_modalities(second, first)

    def test_roberta_preflight_rejects_changed_original_visuals(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "roberta-large"
            model_dir.mkdir()
            features = root / "text-large.pt"
            reference = root / "original.pt"
            features.write_bytes(b"candidate")
            reference.write_bytes(b"reference")
            splits = {split: SimpleNamespace(samples=[dict(sample(split), alignment="uniform")])
                      for split in ("train", "valid", "test")}
            originals = {split: SimpleNamespace(samples=[dict(item) for item in data.samples])
                         for split, data in splits.items()}
            meta = {"splits": {split: 1 for split in splits},
                    "feature_models": {"text": str(model_dir)}}
            args = Namespace(trial="roberta", features=features,
                             reference_features=reference, expected_text_model=model_dir)
            with patch("scripts.mosi_followup.EXPECTED_SPLITS", meta["splits"]), \
                    patch("scripts.mosi_followup.EXPECTED_FINGERPRINT",
                          sample_label_fingerprint(splits)), \
                    patch("scripts.mosi_followup.check_gpu"), \
                    patch("datasets.mosi_extract.load_mosi_extract",
                          side_effect=[(splits, meta), (originals, meta)]):
                report = validate_inputs(args)
            self.assertEqual(report["splits"], meta["splits"])
            self.assertNotEqual(report["features_sha256"],
                                report["reference_features_sha256"])
            originals["test"].samples[0]["visual"] = np.zeros((2, 768), np.float32)
            with patch("scripts.mosi_followup.EXPECTED_SPLITS", meta["splits"]), \
                    patch("scripts.mosi_followup.EXPECTED_FINGERPRINT",
                          sample_label_fingerprint(splits)), \
                    patch("scripts.mosi_followup.check_gpu"), \
                    patch("datasets.mosi_extract.load_mosi_extract",
                          side_effect=[(splits, meta), (originals, meta)]), \
                    self.assertRaisesRegex(ValueError, "original visual was changed"):
                validate_inputs(args)

    def test_masked_modality_stays_absent_from_router_and_loss(self):
        raw = {
            "text": torch.zeros(2, 2, 4), "audio": torch.ones(2, 2, 3),
            "vision": torch.ones(2, 2, 2), "label": torch.tensor([[1.], [-1.]]),
            "text_mask": torch.ones(2, 2, dtype=torch.bool),
            "audio_mask": torch.ones(2, 2, dtype=torch.bool),
            "vision_mask": torch.ones(2, 2, dtype=torch.bool),
            "text_input_ids": torch.tensor([[1, 2, 3], [1, 4, 5]]),
            "text_attention_mask": torch.ones(2, 3, dtype=torch.long),
            "text_word_ids": torch.tensor([[-1, 0, 1], [-1, 0, 1]]),
        }
        for variant in TRIALS["modality"]:
            fusion = BCHRPP(d=8, queries=2, topk=2, text_dim=4, audio_dim=3,
                            vision_dim=2, dropout=0., modality_dropout=0.)
            model = MOSIOnlineText(TinyEncoder(), fusion, train_text=True)
            batch = batch_to_device(raw, torch.device("cpu"), variant.modalities)
            prediction, aux = model(batch["text"], batch["audio"], batch["vision"],
                                    masks=batch["masks"], text_tokens=batch["text_tokens"],
                                    return_aux=True)
            expected = {
                "full": [True, True, True], "no_audio": [True, False, True],
                "no_vision": [True, True, False], "text_only": [True, False, False],
            }[variant.modalities]
            self.assertEqual(aux["modality_available"].tolist(), [expected, expected])
            loss, parts = compute_loss(prediction, batch["label"], aux)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(torch.isfinite(value) for value in parts.values()))
            loss.backward()
            self.assertIsNotNone(model.encoder.embed.weight.grad)
            for name, enabled in (("audio", expected[1]), ("vision", expected[2])):
                if not enabled:
                    grad = getattr(fusion, f"{name}_projection").weight.grad
                    self.assertTrue(grad is None or torch.count_nonzero(grad) == 0)

    def test_optimizer_only_unfreezes_text_for_unfrozen_mode(self):
        for text_mode in ("frozen", "unfrozen"):
            fusion = BCHRPP(d=8, queries=2, topk=2, text_dim=4, audio_dim=3,
                            vision_dim=2)
            model = MOSIOnlineText(TinyEncoder(), fusion, text_mode == "unfrozen")
            args = Namespace(text_mode=text_mode, text_lr=1e-5, lr=3e-4,
                             weight_decay=.01)
            optimizer = make_optimizer(model, args)
            self.assertEqual([group["lr"] for group in optimizer.param_groups],
                             [3e-4, 1e-5] if text_mode == "unfrozen" else [3e-4])
            self.assertEqual(any(p.requires_grad for p in model.encoder.parameters()),
                             text_mode == "unfrozen")

    def test_resume_reuses_complete_runs_and_refuses_changed_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.pt"
            features.write_bytes(b"trusted fixture")
            args = Namespace(trial="text_lr", output_root=root / "trial")
            inputs = {"features": str(features.resolve()), "features_sha256": "a" * 64,
                      "reference_features_sha256": None, "text_model": str(root / "model"),
                      "feature_models": {}, "splits": {}, "sample_label_fingerprint": "b" * 64}
            manifest = make_manifest(args, inputs)
            calls = []

            def fake_train(command, **kwargs):
                calls.append(command)
                checkpoint = Path(command[command.index("--checkpoint") + 1])
                variant = next(v for v in TRIALS["text_lr"] if v.name == checkpoint.parent.name)
                seed = int(command[command.index("--seed") + 1])
                config = variant_config(manifest, variant)
                metrics = {key: .5 for key in METRIC_KEYS}
                torch.save({"version": "bchrpp_rcse_lite_online_text_v1",
                            "dataset": "MOSI", "seed": seed,
                            "features": manifest["features"],
                            "metric_protocol": "metricsTop.py:regression/MOSI",
                            "trial_config": config, "model_state": {"w": torch.ones(1)},
                            "model_config": {"text_mode": variant.text_mode,
                                             "text_model": manifest["text_model"],
                                             "text_lr": variant.text_lr,
                                             "modalities": variant.modalities,
                                             "tokenizer_prefix_space": variant.tokenizer_prefix_space},
                            "epoch": 2, "selection_val_mae": .5}, checkpoint)
                checkpoint.with_suffix(".json").write_text(json.dumps({
                    "dataset": "MOSI", "seed": seed, "features": manifest["features"],
                    "metric_protocol": "metricsTop.py:regression/MOSI",
                    "trial_config": config, "checkpoint": str(checkpoint),
                    "best_epoch": 2, "last_epoch": 3,
                    "validation": metrics, "test": metrics,
                }), encoding="utf8")

            with patch("scripts.mosi_followup.subprocess.run", side_effect=fake_train):
                with contextlib.redirect_stdout(io.StringIO()):
                    run_trial(args, manifest)
                    self.assertEqual(len(calls), 3 * len(SEEDS))
                    run_trial(args, manifest)
                    self.assertEqual(len(calls), 3 * len(SEEDS))
                    changed = dict(manifest, features_sha256="c" * 64)
                    with self.assertRaisesRegex(ValueError, "manifest differs"):
                        run_trial(args, changed)
                    checkpoint = args.output_root / "lr_1e-5" / "mosi_lite_seed222.pt"
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
                    saved["model_config"]["text_lr"] = 9e-5
                    torch.save(saved, checkpoint)
                    with self.assertRaisesRegex(ValueError, "Checkpoint model differs"):
                        run_trial(args, manifest)
            self.assertEqual(len(list(args.output_root.glob("*/summary.csv"))), 3)
            self.assertTrue((args.output_root / "comparison.csv").is_file())

    def test_unfinished_checkpoint_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(trial="modality", output_root=root / "trial")
            inputs = {"features": "feature.pt", "features_sha256": "a" * 64,
                      "reference_features_sha256": None, "text_model": "model",
                      "feature_models": {}, "splits": {}, "sample_label_fingerprint": "b" * 64}
            manifest = make_manifest(args, inputs)
            from scripts.mosi_followup import ensure_manifest
            ensure_manifest(args.output_root, manifest)
            folder = args.output_root / "full"
            folder.mkdir()
            checkpoint = folder / "mosi_lite_seed222.pt"
            checkpoint.write_bytes(b"incomplete")
            with patch("scripts.mosi_followup.subprocess.run") as launch:
                with self.assertRaisesRegex(FileExistsError, "Incomplete checkpoint"):
                    run_trial(args, manifest)
                launch.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"incomplete")


if __name__ == "__main__":
    unittest.main()
