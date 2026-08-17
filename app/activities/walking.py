"""
Walking (Requirement 15, Required Activity 3).

Walking cannot be judged from a single pose -- it requires motion over
time, so this rule reads directly from the PoseSequenceBuffer instead of
a single frame: upright torso + sustained hip/ankle displacement across
several frames.
"""
from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer
from app.activities.base_activity import ActivityDetector

UPRIGHT_TORSO_MAX_DEG = 30.0
MOTION_MIN = 0.02


def condition(pose: Pose, angles: JointAngles, normalized, history: PoseSequenceBuffer) -> bool:
    if len(history) < 5:
        return False
    if angles.torso_angle is not None and angles.torso_angle > UPRIGHT_TORSO_MAX_DEG:
        return False

    motion = history.hip_motion(frames=8)
    if motion is None or motion < MOTION_MIN:
        return False

    # ankles should also be displacing (rules out arm-only or torso-sway motion)
    lv = history.keypoint_velocity("left_ankle", frames=8)
    rv = history.keypoint_velocity("right_ankle", frames=8)
    ankle_motion = [v for v in (lv, rv) if v is not None]
    if ankle_motion and max(ankle_motion) < MOTION_MIN * 0.5:
        return False

    return True


def build_detector(config) -> ActivityDetector:
    return ActivityDetector(
        name="walking",
        condition_fn=condition,
        confirm_frames=max(3, config.candidate_to_confirmed_frames - 2),
        end_grace_frames=config.end_grace_frames,
        min_active_hold_frames=config.confirmed_min_hold_frames,
    )
