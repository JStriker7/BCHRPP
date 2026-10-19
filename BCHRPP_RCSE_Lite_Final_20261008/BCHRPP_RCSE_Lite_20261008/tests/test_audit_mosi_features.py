"""Tests for the pure feature-summary portion of the baseline audit."""

import unittest

import numpy as np

from scripts.audit_mosi_features import summarize_split


class AuditMosiFeaturesTest(unittest.TestCase):
    def test_alignment_masks_and_label_bins(self):
        samples = [
            {
                "label": np.float32(-1.6),
                "alignment": "timestamps",
                "text_valid": np.ones(2),
                "audio_valid": np.array([1, 0]),
                "visual_valid": np.array([0, 0]),
                "sources": {"audio": "a.wav", "visual": None},
            },
            {
                "label": np.float32(0.4),
                "alignment": "uniform",
                "text_valid": np.ones(3),
                "audio_valid": np.ones(3),
                "visual_valid": np.ones(3),
                "sources": {"audio": "b.wav", "visual": "b.mp4"},
            },
        ]
        report = summarize_split(samples)
        self.assertEqual(report["alignment"], {"timestamps": 1, "uniform": 1})
        self.assertEqual(report["words"], {"min": 2, "mean": 2.5, "max": 3})
        self.assertEqual(report["modalities"]["audio"]["valid_positions"], 4)
        self.assertEqual(report["modalities"]["audio"]["partially_invalid_samples"], 1)
        self.assertEqual(report["modalities"]["visual"]["absent_samples"], 1)
        self.assertEqual(report["missing_source_paths"]["visual"], 1)
        self.assertEqual(report["label_bins_rounded"]["-2"], 1)
        self.assertEqual(report["label_bins_rounded"]["0"], 1)


if __name__ == "__main__":
    unittest.main()
