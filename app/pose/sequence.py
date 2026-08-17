"""
Pose sequence buffer (Requirement 5).

Activity recognition should not depend on a single frame. Each tracked
person gets a short rolling buffer (default 30 frames, recommended
range 15-60) of poses + derived features, used for motion direction,
joint-angle change, velocity, stability and duration analysis.
"""
from __future__ import annotations
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional

from app.pose.keypoints import Pose, Keypoint
from app.pose.angles import JointAngles
from app.pose.normalization import estimate_scale


@dataclass
class FrameFeatures:
    pose: Pose
    angles: JointAngles
    normalized: Optional[Dict[str, Keypoint]]
    timestamp: float


class PoseSequenceBuffer:
    """Rolling history of poses for a single tracked person."""

    def __init__(self, maxlen: int = 30):
        self.maxlen = maxlen
        self.buffer: Deque[FrameFeatures] = deque(maxlen=maxlen)

    def add(self, pose: Pose, angles: JointAngles, normalized: Optional[Dict[str, Keypoint]], timestamp: float) -> None:
        self.buffer.append(FrameFeatures(pose=pose, angles=angles, normalized=normalized, timestamp=timestamp))

    def __len__(self) -> int:
        return len(self.buffer)

    def recent(self, n: Optional[int] = None) -> List[FrameFeatures]:
        items = list(self.buffer)
        return items[-n:] if n else items

    def duration_seconds(self) -> float:
        if len(self.buffer) < 2:
            return 0.0
        return self.buffer[-1].timestamp - self.buffer[0].timestamp

    def keypoint_velocity(self, name: str, frames: int = 5) -> Optional[float]:
        """
        Scale-normalized displacement per second of a given keypoint over
        the last `frames` samples, used for walking / sudden-motion
        detection. Deliberately reads RAW pose coordinates (not the
        hip-centered `normalized` pose) and divides by the person's
        torso-length scale, because `normalize_pose` removes translation
        entirely (it's hip-centered) and so can't be used to measure
        motion of the hip itself, or overall body displacement.
        """
        items = self.recent(frames)
        pts = []
        for f in items:
            kp = f.pose.keypoints.get(name)
            if kp is not None and kp.confidence >= 0.3:
                pts.append((kp.x, kp.y, f.timestamp, f.pose))
        if len(pts) < 2:
            return None
        x0, y0, t0, p0 = pts[0]
        x1, y1, t1, p1 = pts[-1]
        dt = t1 - t0
        if dt <= 0:
            return None
        scale = estimate_scale(p1, 0.3) or estimate_scale(p0, 0.3) or 1.0
        dist = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5 / scale
        return dist / dt

    def hip_motion(self, frames: int = 10) -> Optional[float]:
        """
        Average per-frame scale-normalized hip-center displacement -- a
        general motion/stability signal used by standing/walking rules.
        Uses raw pose coordinates for the same reason as keypoint_velocity.
        """
        items = self.recent(frames)
        centers = []
        scales = []
        for f in items:
            p = f.pose
            if p.is_visible("left_hip", 0.3) and p.is_visible("right_hip", 0.3):
                lh, rh = p.get("left_hip"), p.get("right_hip")
                centers.append(((lh.x + rh.x) / 2, (lh.y + rh.y) / 2))
                s = estimate_scale(p, 0.3)
                if s:
                    scales.append(s)
        if len(centers) < 2:
            return None
        avg_scale = (sum(scales) / len(scales)) if scales else 1.0
        total = 0.0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            total += (dx ** 2 + dy ** 2) ** 0.5
        return (total / (len(centers) - 1)) / avg_scale

    def is_stable(self, frames: int = 10, threshold: float = 0.03) -> bool:
        motion = self.hip_motion(frames)
        return motion is not None and motion < threshold
