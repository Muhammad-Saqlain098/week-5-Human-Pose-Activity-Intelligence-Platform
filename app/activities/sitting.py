"""
Sitting (Requirement 15, Required Activity 2).

Rule: hip and knee angles indicate a seated posture (knees bent to
roughly a right angle), torso stays reasonably upright, and the pose
persists (handled by the base state machine).
"""
from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer
from app.activities.base_activity import ActivityDetector

SEATED_KNEE_MIN_DEG = 70.0
SEATED_KNEE_MAX_DEG = 130.0
TORSO_MAX_DEG_FOR_SITTING = 35.0


def condition(pose: Pose, angles: JointAngles, normalized, history: PoseSequenceBuffer) -> bool:
    knee_angles = [a for a in (angles.left_knee, angles.right_knee) if a is not None]
    if not knee_angles:
        return False
    avg_knee = sum(knee_angles) / len(knee_angles)
    if not (SEATED_KNEE_MIN_DEG <= avg_knee <= SEATED_KNEE_MAX_DEG):
        return False

    if angles.torso_angle is not None and angles.torso_angle > TORSO_MAX_DEG_FOR_SITTING:
        return False  # too bent over to be "sitting upright"

    # a seated person's hip should sit noticeably lower relative to shoulders
    # than a standing person -- approximate using normalized hip y vs shoulder y
    if normalized and "left_hip" in normalized and "left_shoulder" in normalized:
        hip_y = normalized["left_hip"].y
        sh_y = normalized["left_shoulder"].y
        if (hip_y - sh_y) < 0.3:
            return False

    return True


def build_detector(config) -> ActivityDetector:
    return ActivityDetector(
        name="sitting",
        condition_fn=condition,
        confirm_frames=config.candidate_to_confirmed_frames,
        end_grace_frames=config.end_grace_frames,
        min_active_hold_frames=config.confirmed_min_hold_frames,
    )
