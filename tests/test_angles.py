import math
from app.pose.keypoints import Keypoint
from app.pose.angles import calculate_angle, compute_joint_angles, torso_angle_from_vertical


def test_calculate_angle_right_angle():
    a = Keypoint(0, -1, 0.9)
    b = Keypoint(0, 0, 0.9)
    c = Keypoint(1, 0, 0.9)
    angle = calculate_angle(a, b, c)
    assert math.isclose(angle, 90.0, abs_tol=0.01)


def test_calculate_angle_straight_line():
    a = Keypoint(0, -1, 0.9)
    b = Keypoint(0, 0, 0.9)
    c = Keypoint(0, 1, 0.9)
    angle = calculate_angle(a, b, c)
    assert math.isclose(angle, 180.0, abs_tol=0.01)


def test_calculate_angle_zero_length_vector_is_safe():
    a = Keypoint(0, 0, 0.9)
    b = Keypoint(0, 0, 0.9)
    c = Keypoint(1, 0, 0.9)
    # degenerate vector (a==b) must not raise
    angle = calculate_angle(a, b, c)
    assert angle == 0.0


def test_standing_pose_has_straight_knees(standing_pose):
    angles = compute_joint_angles(standing_pose, 0.4)
    assert angles.left_knee is not None
    assert angles.left_knee > 150
    assert angles.right_knee > 150


def test_standing_pose_is_upright(standing_pose):
    angles = compute_joint_angles(standing_pose, 0.4)
    assert angles.torso_angle is not None
    assert angles.torso_angle < 20


def test_sitting_pose_has_bent_knees(sitting_pose):
    angles = compute_joint_angles(sitting_pose, 0.4)
    assert 70 <= angles.left_knee <= 130
    assert 70 <= angles.right_knee <= 130


def test_missing_keypoint_returns_none_angle(standing_pose):
    standing_pose.keypoints["left_wrist"].confidence = 0.05
    angles = compute_joint_angles(standing_pose, 0.4)
    assert angles.left_elbow is None
    # right side untouched, should still compute
    assert angles.right_elbow is not None


def test_bending_pose_torso_angle_is_large(bending_pose):
    angle = torso_angle_from_vertical(bending_pose, 0.4)
    assert angle is not None
    assert angle > 35
