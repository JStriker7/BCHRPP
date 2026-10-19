"""Small deterministic checks for the face experiment's unchanged alignment."""

import unittest
from types import SimpleNamespace

import numpy as np

from scripts.face_crop_ops import crop_first_face, pool_frame_features


class FaceCropOpsTest(unittest.TestCase):
    def test_crop_clamps_bounds_and_falls_back_when_detection_fails(self):
        frame = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        detection = SimpleNamespace(
            categories=[SimpleNamespace(score=0.9)],
            bounding_box=SimpleNamespace(origin_x=-2, origin_y=1, width=5, height=2),
        )
        crop, found = crop_first_face(frame, [detection], min_score=0.8)
        self.assertTrue(found)
        np.testing.assert_array_equal(crop, frame[1:3, 0:3])
        original, found = crop_first_face(frame, [], min_score=0.8)
        self.assertFalse(found)
        self.assertIs(original, frame)
        detection.categories[0].score = 0.7
        original, found = crop_first_face(frame, [detection], min_score=0.8)
        self.assertFalse(found)
        self.assertIs(original, frame)
        detection.categories[0].score = 0.9
        detection.bounding_box.width = 0
        original, found = crop_first_face(frame, [detection], min_score=0.8)
        self.assertFalse(found)
        self.assertIs(original, frame)

    def test_pooling_uses_original_windows_and_nearest_fallback(self):
        embeddings = np.asarray([[1, 2], [3, 4], [5, 6]], dtype=np.float32)
        times = np.asarray([0, 0.5, 1], dtype=np.float32)
        intervals = [(0, 0.6), (0.6, 0.8), (0.8, 1.2)]
        result = pool_frame_features(embeddings, times, intervals)
        np.testing.assert_array_equal(result, [[2, 3], [3, 4], [5, 6]])


if __name__ == "__main__":
    unittest.main()
