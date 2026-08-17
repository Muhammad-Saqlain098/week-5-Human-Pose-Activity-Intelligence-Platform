"""
Pose normalization (Requirement 7).

Raw pixel coordinates vary with camera distance, resolution, person height,
and position in frame, which would otherwise make every threshold
person- and camera-specific. We normalize by:

  1. Centering all keypoints on the hip midpoint (position-invariant).
  2. Scaling by torso length (shoulder-midpoint to hip-midpoint distance),
     falling back to shoulder width if the torso can't be measured
     (scale-invariant, largely distance/resolution-invariant).

This keeps activity thresholds (e.g. "wrist above shoulder by X units")
comparable across people, cameras and resolutions.
"""
from __future__ import annotations
import math
from typing import Dict, Optional

from app.pose.keypoints import Pose, Keypoint, COCO_KEYPOINTS


def _mid(a: Keypoint, b: Keypoint):
    return (a.x + b.x) / 2, (a.y + b.y) / 2


def estimate_scale(pose: Pose, min_conf: float) -> Optional[float]:
    """
    Torso-length (falls back to shoulder-width) scale estimate for a pose,
    in raw pixel units. Used to normalize *velocities* and other
    translation-sensitive measurements without discarding position info
    the way `normalize_pose` (which is hip-centered) necessarily does.
    """
    if not (pose.is_visible("left_hip", min_conf) and pose.is_visible("right_hip", min_conf)):
        return None
    hip_mid_x, hip_mid_y = _mid(pose.get("left_hip"), pose.get("right_hip"))
    scale = None
    if pose.is_visible("left_shoulder", min_conf) and pose.is_visible("right_shoulder", min_conf):
        sh_mid_x, sh_mid_y = _mid(pose.get("left_shoulder"), pose.get("right_shoulder"))
        scale = math.hypot(sh_mid_x - hip_mid_x, sh_mid_y - hip_mid_y)
    if not scale or scale < 1e-6:
        if pose.is_visible("left_shoulder", min_conf) and pose.is_visible("right_shoulder", min_conf):
            ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
            scale = math.hypot(ls.x - rs.x, ls.y - rs.y)
    return scale if scale and scale > 1e-6 else None


def normalize_pose(pose: Pose, min_conf: float) -> Optional[Dict[str, Keypoint]]:
    """
    Returns a dict of normalized keypoints (hip-centered, torso-scaled),
    or None if there isn't enough visible anatomy to normalize reliably.
    """
    if not (pose.is_visible("left_hip", min_conf) and pose.is_visible("right_hip", min_conf)):
        return None

    hip_mid_x, hip_mid_y = _mid(pose.get("left_hip"), pose.get("right_hip"))

    scale = None
    if pose.is_visible("left_shoulder", min_conf) and pose.is_visible("right_shoulder", min_conf):
        sh_mid_x, sh_mid_y = _mid(pose.get("left_shoulder"), pose.get("right_shoulder"))
        scale = math.hypot(sh_mid_x - hip_mid_x, sh_mid_y - hip_mid_y)

    if not scale or scale < 1e-6:
        # fall back to shoulder width
        if pose.is_visible("left_shoulder", min_conf) and pose.is_visible("right_shoulder", min_conf):
            ls, rs = pose.get("left_shoulder"), pose.get("right_shoulder")
            scale = math.hypot(ls.x - rs.x, ls.y - rs.y)

    if not scale or scale < 1e-6:
        return None

    normalized: Dict[str, Keypoint] = {}
    for name in COCO_KEYPOINTS:
        kp = pose.keypoints.get(name)
        if kp is None:
            continue
        nx = (kp.x - hip_mid_x) / scale
        ny = (kp.y - hip_mid_y) / scale
        normalized[name] = Keypoint(nx, ny, kp.confidence)
    return normalized
