"""
Person tracking (Requirement 4).

YOLoPoseEstimator.infer() already returns persistent IDs via YOLO's
built-in ByteTrack/BoT-SORT when use_tracking=True, which is the
recommended path. This module provides a simple, dependency-free IOU
tracker as a documented fallback for `tracker: simple_iou` in config,
e.g. when running with a pose backend that has no native tracker.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from app.pose.keypoints import Pose


def _iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class SimpleIOUTracker:
    """Greedy IOU-based tracker: matches new detections to the previous
    frame's tracks by highest IOU, assigning new IDs to unmatched boxes."""

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: Dict[int, Tuple[float, float, float, float]] = {}
        self._missed: Dict[int, int] = {}
        self._next_id = 1

    def update(self, poses: List[Pose]) -> List[Pose]:
        unmatched_tracks = set(self._tracks.keys())
        assigned: List[Pose] = []

        for pose in poses:
            best_id, best_iou = None, 0.0
            for tid in unmatched_tracks:
                score = _iou(pose.bbox, self._tracks[tid])
                if score > best_iou:
                    best_iou, best_id = score, tid
            if best_id is not None and best_iou >= self.iou_threshold:
                pose.person_id = best_id
                self._tracks[best_id] = pose.bbox
                self._missed[best_id] = 0
                unmatched_tracks.discard(best_id)
            else:
                new_id = self._next_id
                self._next_id += 1
                pose.person_id = new_id
                self._tracks[new_id] = pose.bbox
                self._missed[new_id] = 0
            assigned.append(pose)

        for tid in unmatched_tracks:
            self._missed[tid] = self._missed.get(tid, 0) + 1
        expired = [tid for tid, m in self._missed.items() if m > self.max_missed]
        for tid in expired:
            self._tracks.pop(tid, None)
            self._missed.pop(tid, None)

        return assigned
