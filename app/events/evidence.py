"""
Evidence capture (Requirement 17, Stage 12).

Saves full frames and person crops to disk for important activities.
Uses OpenCV when available; degrades to a no-op with a logged warning
if cv2 or the frame is unavailable, so the rest of the pipeline never
crashes because evidence saving failed (Requirement 39).
"""
from __future__ import annotations
import os
import time
import logging
from typing import Optional, Tuple

logger = logging.getLogger("evidence")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


class EvidenceStore:
    def __init__(self, base_dir: str = "evidence"):
        self.base_dir = base_dir
        for sub in ("fall", "bending", "alerts"):
            os.makedirs(os.path.join(base_dir, sub), exist_ok=True)

    def _path(self, category: str, person_id: int, label: str) -> str:
        ts = int(time.time() * 1000)
        fname = f"person{person_id}_{label}_{ts}.jpg"
        return os.path.join(self.base_dir, category, fname)

    def save_frame(self, frame, category: str, person_id: int, label: str,
                    bbox: Optional[Tuple[float, float, float, float]] = None) -> Optional[str]:
        if not _HAS_CV2 or frame is None:
            logger.warning("Evidence save skipped (no frame / cv2 unavailable): %s person=%s", label, person_id)
            return None
        try:
            path = self._path(category, person_id, label)
            cv2.imwrite(path, frame)
            return path
        except Exception as e:  # pragma: no cover
            logger.error("Failed to save evidence frame: %s", e)
            return None

    def save_person_crop(self, frame, bbox, category: str, person_id: int, label: str) -> Optional[str]:
        if not _HAS_CV2 or frame is None:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                return None
            crop = frame[y1:y2, x1:x2]
            path = self._path(category, person_id, f"{label}_crop")
            cv2.imwrite(path, crop)
            return path
        except Exception as e:  # pragma: no cover
            logger.error("Failed to save evidence crop: %s", e)
            return None

    def save_fall_sequence(self, pre_frame, event_frame, post_frame, person_id: int) -> dict:
        """Requirement 17: for fall events, save pre/event/post-event frames where practical."""
        paths = {}
        if pre_frame is not None:
            paths["pre"] = self.save_frame(pre_frame, "fall", person_id, "pre")
        paths["event"] = self.save_frame(event_frame, "fall", person_id, "event")
        if post_frame is not None:
            paths["post"] = self.save_frame(post_frame, "fall", person_id, "post")
        return paths
