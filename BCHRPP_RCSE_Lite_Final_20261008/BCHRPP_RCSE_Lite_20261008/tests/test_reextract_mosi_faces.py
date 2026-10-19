"""Input and cache guardrails for the isolated face-crop experiment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.reextract_mosi_faces import source_stamp, validate_input


def original_features():
    models = {"visual": "/local/clip", "text": "/local/deberta", "audio": "/local/clap"}
    data = {"_metadata": {
        "format": "bhgm_sent_crc_mosi_v1",
        "dimensions": {"text": 1024, "audio": 512, "visual": 768},
        "models": models, "splits": {"train": 1, "valid": 1, "test": 1},
    }}
    for split in ("train", "valid", "test"):
        data[split] = [{
            "id": f"{split}_1", "words": ["hello"], "alignment": "uniform",
            "feature_models": models, "visual": np.zeros((1, 768), dtype=np.float32),
            "visual_valid": np.ones(1, dtype=np.float32),
            "sources": {"audio": f"{split}.wav", "visual": f"{split}.mp4"},
        }]
    return data


class FaceFeatureInputTest(unittest.TestCase):
    def test_valid_original_and_no_recropping(self):
        data = original_features()
        self.assertIs(validate_input(data), data["_metadata"])
        data["_metadata"]["face_crop"] = {}
        with self.assertRaisesRegex(ValueError, "already face-cropped"):
            validate_input(data)

    def test_rejects_nonuniform_or_changed_visual_model(self):
        data = original_features()
        data["train"][0]["alignment"] = "timestamps"
        with self.assertRaisesRegex(ValueError, "timestamp alignment"):
            validate_input(data)
        data["train"][0]["alignment"] = "uniform"
        data["train"][0]["feature_models"] = {"visual": "other"}
        with self.assertRaisesRegex(ValueError, "identifiers differ"):
            validate_input(data)

    def test_source_stamp_changes_with_frame_content_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "001.png").write_bytes(b"first")
            before = source_stamp(root)
            (root / "002.png").write_bytes(b"second")
            self.assertNotEqual(before, source_stamp(root))


if __name__ == "__main__":
    unittest.main()
