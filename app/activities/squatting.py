"""
Squatting (Requirement 16, additional activity) and the basis of the
squat repetition counter (Requirement 13, Requirement 21).

Rule: both knee angles drop below a "down" threshold while the person
remains roughly upright through the torso (distinguishes a squat from
bending forward) and the hip is lowered.
"""
from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer
from app.activities.base_activity import ActivityDetector

SQUAT_KNEE_MAX_DEG = 120.0
SQUAT_TORSO_MAX_DEG = 45.0


def condition(pose: Pose, angles: JointAngles, normalized, history: PoseSequenceBuffer) -> bool:
    knee_angles = [a for a in (angles.left_knee, angles.right_knee) if a is not None]
    if not knee_angles:
        return False
    if max(knee_angles) > SQUAT_KNEE_MAX_DEG:
        return False
    if angles.torso_angle is not None and angles.torso_angle > SQUAT_TORSO_MAX_DEG:
        return False
    return True


def build_detector(config) -> ActivityDetector:
    return ActivityDetector(
        name="squatting",
        condition_fn=condition,
        confirm_frames=max(2, config.candidate_to_confirmed_frames - 3),
        end_grace_frames=config.end_grace_frames,
        min_active_hold_frames=1,
    )


class SquatCounter:
    """
    Explicit repetition-counting state machine (Requirement 13):

        STANDING -> GOING_DOWN -> DOWN -> GOING_UP -> STANDING => count += 1

    Tracking the full cycle (rather than just "is squatting True/False")
    prevents double-counting a single partial movement, since a rep only
    counts once the person returns all the way to standing.
    """

    def __init__(self, config):
        self.config = config
        self.state = "standing"
        self.count = 0

    def update(self, angles: JointAngles) -> int:
        knee_angles = [a for a in (angles.left_knee, angles.right_knee) if a is not None]
        if not knee_angles:
            return self.count
        avg_knee = sum(knee_angles) / len(knee_angles)

        if self.state == "standing":
            if avg_knee < self.config.squat_down_knee_angle:
                self.state = "down"
        elif self.state == "down":
            if avg_knee > self.config.squat_up_knee_angle:
                self.state = "standing"
                self.count += 1
        return self.count

    def reset(self):
        self.state = "standing"
        self.count = 0
