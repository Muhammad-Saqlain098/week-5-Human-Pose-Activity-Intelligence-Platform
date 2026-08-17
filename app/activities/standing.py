"""
Standing (Requirement 15, Required Activity 1).

Rule: torso is upright (small torso_angle from vertical), knees are close
to straight (large knee angle), and the person is not moving much.
"""
from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer
from app.activities.base_activity import ActivityDetector

UPRIGHT_TORSO_MAX_DEG = 20.0
STRAIGHT_KNEE_MIN_DEG = 150.0
MOTION_MAX = 0.03


def condition(pose: Pose, angles: JointAngles, normalized, history: PoseSequenceBuffer) -> bool:
    if angles.torso_angle is None:
        return False
    if angles.torso_angle > UPRIGHT_TORSO_MAX_DEG:
        return False

    knee_angles = [a for a in (angles.left_knee, angles.right_knee) if a is not None]
    if not knee_angles:
        return False
    if min(knee_angles) < STRAIGHT_KNEE_MIN_DEG:
        return False

    motion = history.hip_motion(frames=10)
    if motion is not None and motion > MOTION_MAX:
        return False  # moving too much to be simply "standing" (likely walking)

    return True


def build_detector(config) -> ActivityDetector:
    return ActivityDetector(
        name="standing",
        condition_fn=condition,
        confirm_frames=config.candidate_to_confirmed_frames,
        end_grace_frames=config.end_grace_frames,
        min_active_hold_frames=config.confirmed_min_hold_frames,
    )
