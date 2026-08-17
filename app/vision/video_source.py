"""
Video source abstraction (Section 6): uploaded video, webcam, and
(bonus) RTSP stream / multiple sources, all behind one interface.
"""
from __future__ import annotations
import logging
from typing import Optional, Iterator, Tuple

logger = logging.getLogger("video_source")

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False


class VideoSource:
    """
    Wraps cv2.VideoCapture for:
      - webcam:      source="0" (int index as string) or int 0
      - video file:  source="sample_videos/demo.mp4"
      - RTSP stream: source="rtsp://..."
    """

    def __init__(self, source: str, width: Optional[int] = None, height: Optional[int] = None):
        if not _HAS_CV2:
            raise RuntimeError(
                "OpenCV (cv2) is not installed. Run: pip install opencv-python"
            )
        self.source_raw = source
        cap_source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(cap_source)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

    def fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps if fps and fps > 0 else 30.0

    def frames(self) -> Iterator[Tuple[int, "cv2.Mat"]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                logger.info("Video source ended or disconnected: %s", self.source_raw)
                break
            yield idx, frame
            idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()
