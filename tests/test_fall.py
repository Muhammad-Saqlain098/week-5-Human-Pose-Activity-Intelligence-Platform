from app.pose.angles import compute_joint_angles
from app.pose.normalization import normalize_pose
from app.pose.sequence import PoseSequenceBuffer
from app.activities import fall


def test_fall_condition_true_for_lying_pose(lying_pose):
    angles = compute_joint_angles(lying_pose, 0.4)
    norm = normalize_pose(lying_pose, 0.4)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(lying_pose, angles, norm, 0.0)
    assert fall.condition(lying_pose, angles, norm, history) is True


def test_fall_condition_false_for_standing_pose(standing_pose):
    angles = compute_joint_angles(standing_pose, 0.4)
    norm = normalize_pose(standing_pose, 0.4)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(standing_pose, angles, norm, 0.0)
    assert fall.condition(standing_pose, angles, norm, history) is False


def test_fall_condition_false_for_sitting_pose(sitting_pose):
    # A key requirement: sitting must NOT be mistaken for a fall.
    angles = compute_joint_angles(sitting_pose, 0.4)
    norm = normalize_pose(sitting_pose, 0.4)
    history = PoseSequenceBuffer(maxlen=30)
    history.add(sitting_pose, angles, norm, 0.0)
    assert fall.condition(sitting_pose, angles, norm, history) is False


def test_sudden_drop_detection():
    from tests.conftest import make_pose
    history = PoseSequenceBuffer(maxlen=30)
    # standing for a few frames, then hip suddenly drops (normalized) -> sudden fall onset
    for i in range(3):
        pose = make_pose()
        angles = compute_joint_angles(pose, 0.4)
        norm = normalize_pose(pose, 0.4)
        history.add(pose, angles, norm, float(i))
    dropped_pose = make_pose(overrides={
        "left_hip": (90, 200, 0.9), "right_hip": (110, 200, 0.9),
    })
    angles = compute_joint_angles(dropped_pose, 0.4)
    norm = normalize_pose(dropped_pose, 0.4)
    history.add(dropped_pose, angles, norm, 3.0)
    assert fall.had_sudden_drop(history) is True


def test_no_sudden_drop_when_stationary(standing_pose):
    history = PoseSequenceBuffer(maxlen=30)
    angles = compute_joint_angles(standing_pose, 0.4)
    norm = normalize_pose(standing_pose, 0.4)
    for i in range(6):
        history.add(standing_pose, angles, norm, float(i))
    assert fall.had_sudden_drop(history) is False


def test_fall_lifecycle_confirm_and_resolve():
    event = fall.FallEvent(event_id="abc123", person_id=1, start_time=0.0, confidence=0.3)
    assert event.status == "possible_fall"
    event.confirm(1.0)
    assert event.status == "fall_confirmed"
    event.activate_alert(1.0)
    assert event.status == "alert_active"
    event.resolve(5.0)
    assert event.status == "resolved"
    assert event.end_time == 5.0
