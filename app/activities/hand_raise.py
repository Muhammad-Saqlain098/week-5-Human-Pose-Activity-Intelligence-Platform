"""
Hand Raised (Requirement 15, Required Activity 5).

Rule: wrist y-coordinate is above (numerically less than, since image
coordinates increase downward) the shoulder y-coordinate by a margin,
with sufficient keypoint confidence, held for several frames.
"""
from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer
from app.activities.base_activity import ActivityDetector

RAISE_MARGIN_NORM = 0.15  # in normalized (torso-scaled) units


def _hand_raised_side(normalized, side: str) -> bool:
    if not normalized:
        return False
    wrist = normalized.get(f"{side}_wrist")
    shoulder = normalized.get(f"{side}_shoulder")
    if wrist is None or shoulder is None:
        return False
    return (shoulder.y - wrist.y) > RAISE_MARGIN_NORM  # wrist above shoulder


def condition(pose: Pose, angles: JointAngles, normalized, history: PoseSequenceBuffer) -> bool:
    min_conf = 0.4
    left_ok = pose.is_visible("left_wrist", min_conf) and pose.is_visible("left_shoulder", min_conf)
    right_ok = pose.is_visible("right_wrist", min_conf) and pose.is_visible("right_shoulder", min_conf)
    if not (left_ok or right_ok):
        return False

    left_raised = left_ok and _hand_raised_side(normalized, "left")
    right_raised = right_ok and _hand_raised_side(normalized, "right")
    return left_raised or right_raised


def build_detector(config) -> ActivityDetector:
    return ActivityDetector(
        name="hand_raised",
        condition_fn=condition,
        confirm_frames=config.hand_raise_min_frames,
        end_grace_frames=3,
        min_active_hold_frames=1,
    )
