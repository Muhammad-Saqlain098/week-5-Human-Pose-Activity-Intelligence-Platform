from app.pose.angles import compute_joint_angles
from app.pose.normalization import normalize_pose
from app.pose.sequence import PoseSequenceBuffer
from app.activities import standing, sitting, hand_raise, bending, squatting, walking


def _features(pose, min_conf=0.4):
    angles = compute_joint_angles(pose, min_conf)
    normalized = normalize_pose(pose, min_conf)
    return angles, normalized


def test_standing_rule_true_for_standing_pose(standing_pose):
    angles, norm = _features(standing_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(standing_pose, angles, norm, 0.0)
    assert standing.condition(standing_pose, angles, norm, history) is True


def test_standing_rule_false_for_sitting_pose(sitting_pose):
    angles, norm = _features(sitting_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(sitting_pose, angles, norm, 0.0)
    assert standing.condition(sitting_pose, angles, norm, history) is False


def test_sitting_rule_true_for_sitting_pose(sitting_pose):
    angles, norm = _features(sitting_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(sitting_pose, angles, norm, 0.0)
    assert sitting.condition(sitting_pose, angles, norm, history) is True


def test_sitting_rule_false_for_standing_pose(standing_pose):
    angles, norm = _features(standing_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(standing_pose, angles, norm, 0.0)
    assert sitting.condition(standing_pose, angles, norm, history) is False


def test_hand_raised_rule_true(hand_raised_pose):
    angles, norm = _features(hand_raised_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(hand_raised_pose, angles, norm, 0.0)
    assert hand_raise.condition(hand_raised_pose, angles, norm, history) is True


def test_hand_raised_rule_false_for_standing(standing_pose):
    angles, norm = _features(standing_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(standing_pose, angles, norm, 0.0)
    assert hand_raise.condition(standing_pose, angles, norm, history) is False


def test_bending_rule_true_for_bending_pose(bending_pose):
    angles, norm = _features(bending_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(bending_pose, angles, norm, 0.0)
    assert bending.condition(bending_pose, angles, norm, history) is True


def test_squatting_rule_true_for_squat_pose(squat_down_pose):
    angles, norm = _features(squat_down_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(squat_down_pose, angles, norm, 0.0)
    assert squatting.condition(squat_down_pose, angles, norm, history) is True


def test_squatting_rule_false_for_standing(standing_pose):
    angles, norm = _features(standing_pose)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(standing_pose, angles, norm, 0.0)
    assert squatting.condition(standing_pose, angles, norm, history) is False


def test_walking_requires_multi_frame_motion(standing_pose):
    # a single static frame (no motion history) must NOT be classified as walking
    angles, norm = _features(standing_pose)
    history = PoseSequenceBuffer(maxlen=30)
    for i in range(6):
        history.add(standing_pose, angles, norm, float(i))  # identical pose repeated -> no motion
    assert walking.condition(standing_pose, angles, norm, history) is False


def test_walking_true_with_hip_and_ankle_motion():
    from tests.conftest import make_pose
    history = PoseSequenceBuffer(maxlen=30)
    pose = None
    angles = norm = None
    for i in range(8):
        shift = i * 6  # simulate the person moving across frames
        pose = make_pose(overrides={
            "left_hip": (90 + shift, 130, 0.9), "right_hip": (110 + shift, 130, 0.9),
            "left_ankle": (90 + shift * 1.2, 220, 0.9), "right_ankle": (110 + shift * 1.2, 220, 0.9),
            "left_knee": (90 + shift, 175, 0.9), "right_knee": (110 + shift, 175, 0.9),
        })
        angles = compute_joint_angles(pose, 0.4)
        norm = normalize_pose(pose, 0.4)
        history.add(pose, angles, norm, float(i))
    assert walking.condition(pose, angles, norm, history) is True
