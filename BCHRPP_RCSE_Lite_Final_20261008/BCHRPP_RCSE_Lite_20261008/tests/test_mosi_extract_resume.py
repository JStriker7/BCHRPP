"""Runner resumes only matching, completed per-seed JSON results."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.run_mosi_extract_ablations import METRIC_KEYS, load_run_result, main


class MosiExtractResumeTest(unittest.TestCase):
    def test_resume_refuses_to_overwrite_unfinalized_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.pt"
            features.touch()
            output = root / "runs"
            output.mkdir()
            checkpoint = output / "mosi_lite_seed222.pt"
            checkpoint.write_bytes(b"existing checkpoint")
            argv = ["runner", "--features", str(features), "--output-dir", str(output),
                    "--modes", "lite", "--seeds", "222,128", "--resume"]

            with patch.object(sys, "argv", argv), \
                    patch("subprocess.run") as run, \
                    self.assertRaisesRegex(FileExistsError, "Incomplete run"):
                main()
            run.assert_not_called()
            self.assertEqual(checkpoint.read_bytes(), b"existing checkpoint")
            self.assertFalse((output / "summary.csv").exists())

    def test_reuses_recovered_result_and_writes_summary_without_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.pt"
            features.touch()
            output = root / "runs"
            output.mkdir()
            checkpoint = output / "mosi_lite_seed222.pt"
            checkpoint.touch()
            recovered = output / "mosi_lite_seed222_recovered.pt"
            recovered.touch()
            metrics = {key: 0.5 for key in METRIC_KEYS}
            result = {
                "dataset": "MOSI", "features": str(features), "seed": 222,
                "metric_protocol": "metricsTop.py:regression/MOSI",
                "checkpoint": str(recovered), "best_epoch": 8, "last_epoch": 23,
                "validation": metrics, "test": metrics,
            }
            sidecar = checkpoint.with_suffix(".json")
            sidecar.write_text(json.dumps(result), encoding="utf8")
            (output / "results.csv").write_text("partial\n", encoding="utf8")
            argv = ["runner", "--features", str(features), "--output-dir", str(output),
                    "--modes", "lite", "--seeds", "222,128", "--resume"]

            def finish_seed_128(cmd, **kwargs):
                new_checkpoint = Path(cmd[cmd.index("--checkpoint") + 1])
                new_checkpoint.touch()
                new_result = dict(result, seed=128, checkpoint=str(new_checkpoint))
                new_checkpoint.with_suffix(".json").write_text(
                    json.dumps(new_result), encoding="utf8"
                )

            with patch.object(sys, "argv", argv), \
                    patch("subprocess.run", side_effect=finish_seed_128) as run:
                main()
            run.assert_called_once()
            self.assertIn("test_Mult_acc_7_mean", (output / "summary.csv").read_text(encoding="utf8"))
            self.assertIn("lite,2,", (output / "summary.csv").read_text(encoding="utf8"))

            result["features"] = str(root / "wrong.pt")
            sidecar.write_text(json.dumps(result), encoding="utf8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_run_result(sidecar, checkpoint, features, 222)


if __name__ == "__main__":
    unittest.main()
