from app.pose.normalization import normalize_pose


def test_normalize_pose_centers_hip_at_origin(standing_pose):
    norm = normalize_pose(standing_pose, 0.4)
    assert norm is not None
    hip_mid_x = (norm["left_hip"].x + norm["right_hip"].x) / 2
    hip_mid_y = (norm["left_hip"].y + norm["right_hip"].y) / 2
    assert abs(hip_mid_x) < 1e-6
    assert abs(hip_mid_y) < 1e-6


def test_normalize_pose_scale_invariant():
    from tests.conftest import make_pose
    small = make_pose()
    big_overrides = {}
    for name, kp in small.keypoints.items():
        big_overrides[name] = (kp.x * 2, kp.y * 2, kp.confidence)
    big = make_pose(overrides=big_overrides)

    norm_small = normalize_pose(small, 0.4)
    norm_big = normalize_pose(big, 0.4)
    # after normalization, a 2x scaled-up person should look ~identical
    assert abs(norm_small["left_shoulder"].x - norm_big["left_shoulder"].x) < 1e-6
    assert abs(norm_small["left_shoulder"].y - norm_big["left_shoulder"].y) < 1e-6


def test_normalize_pose_missing_hips_returns_none(standing_pose):
    standing_pose.keypoints["left_hip"].confidence = 0.05
    standing_pose.keypoints["right_hip"].confidence = 0.05
    assert normalize_pose(standing_pose, 0.4) is None
