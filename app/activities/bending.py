"""
Bending (Requirement 16, additional activity) + basis for the
Unsafe Bending ergonomic warning (Requirement 22).

Rule: torso angle from vertical exceeds a threshold while the person
remains standing on roughly straight-ish legs (distinguishes bending
from sitting, where the hip is lowered and knees are strongly bent).
"""
from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer
from app.activities.base_activity import ActivityDetector

BEND_TORSO_MIN_DEG = 35.0
BEND_KNEE_MIN_DEG = 120.0  # legs not deeply bent -- rules out squatting/sitting


def condition(pose: Pose, angles: JointAngles, normalized, history: PoseSequenceBuffer) -> bool:
    if angles.torso_angle is None or angles.torso_angle < BEND_TORSO_MIN_DEG:
        return False
    knee_angles = [a for a in (angles.left_knee, angles.right_knee) if a is not None]
    if knee_angles and min(knee_angles) < BEND_KNEE_MIN_DEG:
        return False  # legs are bent too -- more likely squat/sit
    return True


def build_detector(config) -> ActivityDetector:
    return ActivityDetector(
        name="bending",
        condition_fn=condition,
        confirm_frames=config.candidate_to_confirmed_frames,
        end_grace_frames=config.end_grace_frames,
        min_active_hold_frames=config.confirmed_min_hold_frames,
    )
