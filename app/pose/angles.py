"""
Reusable joint-angle calculation utilities (Requirement 6).

All angles are returned in degrees, in the range [0, 180].
Every function tolerates missing/low-confidence keypoints by returning
None instead of raising, so callers can degrade gracefully
(Requirement 3 / Requirement 39).
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict

from app.pose.keypoints import Pose, Keypoint


def calculate_angle(a: Keypoint, b: Keypoint, c: Keypoint) -> float:
    """
    Angle ABC (the angle at vertex B) formed by points A-B-C, in degrees.
    This is the single reusable primitive used for every joint angle
    instead of writing a bespoke formula per joint.
    """
    ax, ay = a.x - b.x, a.y - b.y
    cx, cy = c.x - b.x, c.y - b.y
    dot = ax * cx + ay * cy
    mag_a = math.hypot(ax, ay)
    mag_c = math.hypot(cx, cy)
    if mag_a == 0 or mag_c == 0:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (mag_a * mag_c)))
    return math.degrees(math.acos(cos_angle))


def _safe_angle(pose: Pose, a: str, b: str, c: str, min_conf: float) -> Optional[float]:
    if not (pose.is_visible(a, min_conf) and pose.is_visible(b, min_conf) and pose.is_visible(c, min_conf)):
        return None
    return calculate_angle(pose.get(a), pose.get(b), pose.get(c))


def torso_angle_from_vertical(pose: Pose, min_conf: float) -> Optional[float]:
    """
    Angle of the torso (shoulder-midpoint to hip-midpoint line) relative to
    vertical (0 deg = perfectly upright, 90 deg = horizontal). Used for
    posture / bending / fall analysis.
    """
    required = ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]
    if any(not pose.is_visible(k, min_conf) for k in required):
        return None
    sh = pose.get("left_shoulder")
    sh2 = pose.get("right_shoulder")
    hp = pose.get("left_hip")
    hp2 = pose.get("right_hip")
    mid_shoulder = ((sh.x + sh2.x) / 2, (sh.y + sh2.y) / 2)
    mid_hip = ((hp.x + hp2.x) / 2, (hp.y + hp2.y) / 2)
    dx = mid_shoulder[0] - mid_hip[0]
    dy = mid_shoulder[1] - mid_hip[1]
    # vertical reference vector is (0, -1) in image coordinates (up)
    angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-9))
    return angle


def hip_angle(pose: Pose, min_conf: float, side: str = "left") -> Optional[float]:
    """Angle at the hip between shoulder-hip-knee (trunk vs thigh)."""
    return _safe_angle(pose, f"{side}_shoulder", f"{side}_hip", f"{side}_knee", min_conf)


@dataclass
class JointAngles:
    left_elbow: Optional[float] = None
    right_elbow: Optional[float] = None
    left_knee: Optional[float] = None
    right_knee: Optional[float] = None
    hip_angle: Optional[float] = None          # average of left/right hip angle
    torso_angle: Optional[float] = None        # degrees from vertical

    def as_dict(self) -> Dict[str, Optional[float]]:
        return {
            "left_elbow": self.left_elbow, "right_elbow": self.right_elbow,
            "left_knee": self.left_knee, "right_knee": self.right_knee,
            "hip_angle": self.hip_angle, "torso_angle": self.torso_angle,
        }


def compute_joint_angles(pose: Pose, min_conf: float) -> JointAngles:
    """Compute the full set of minimum-required joint angles (Requirement 6)."""
    left_elbow = _safe_angle(pose, "left_shoulder", "left_elbow", "left_wrist", min_conf)
    right_elbow = _safe_angle(pose, "right_shoulder", "right_elbow", "right_wrist", min_conf)
    left_knee = _safe_angle(pose, "left_hip", "left_knee", "left_ankle", min_conf)
    right_knee = _safe_angle(pose, "right_hip", "right_knee", "right_ankle", min_conf)

    lh = hip_angle(pose, min_conf, "left")
    rh = hip_angle(pose, min_conf, "right")
    if lh is not None and rh is not None:
        hip = (lh + rh) / 2
    else:
        hip = lh if lh is not None else rh

    torso = torso_angle_from_vertical(pose, min_conf)

    return JointAngles(
        left_elbow=left_elbow, right_elbow=right_elbow,
        left_knee=left_knee, right_knee=right_knee,
        hip_angle=hip, torso_angle=torso,
    )
