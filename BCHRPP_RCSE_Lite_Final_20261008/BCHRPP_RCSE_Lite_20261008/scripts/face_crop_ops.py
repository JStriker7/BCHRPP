"""Pure frame-cropping and word-pooling operations for the MOSI face trial."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def crop_first_face(frame: np.ndarray, detections: Sequence[Any], min_score: float) -> tuple[np.ndarray, bool]:
    """Match the reference detector's first-face/score rule; preserve failed frames."""
    if not detections or not detections[0].categories:
        return frame, False
    face = detections[0]
    if face.categories[0].score < min_score:
        return frame, False
    box = face.bounding_box
    height, width = frame.shape[:2]
    x1, y1 = max(0, int(box.origin_x)), max(0, int(box.origin_y))
    x2 = min(width, int(box.origin_x + box.width))
    y2 = min(height, int(box.origin_y + box.height))
    if x2 <= x1 or y2 <= y1:
        return frame, False
    return np.ascontiguousarray(frame[y1:y2, x1:x2]), True


def pool_frame_features(
    frame_features: np.ndarray,
    frame_times: np.ndarray,
    intervals: Sequence[tuple[float, float]],
) -> np.ndarray:
    """Use the original extractor's mean/nearest-frame word assignment."""
    if len(frame_features) == 0 or len(frame_features) != len(frame_times):
        raise ValueError("frame embeddings and timestamps must have matching nonzero lengths")
    result = np.zeros((len(intervals), frame_features.shape[-1]), dtype=np.float32)
    for index, (start, end) in enumerate(intervals):
        selected = np.flatnonzero((frame_times >= start) & (frame_times < end))
        if selected.size == 0:
            selected = np.asarray([int(np.abs(frame_times - (start + end) / 2).argmin())])
        result[index] = frame_features[selected].mean(axis=0)
    return result
