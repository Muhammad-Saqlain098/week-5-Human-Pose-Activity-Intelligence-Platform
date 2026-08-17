from app.pose.keypoints import Pose, Keypoint, COCO_KEYPOINTS


def test_pose_is_visible(standing_pose):
    assert standing_pose.is_visible("left_shoulder", 0.5)


def test_pose_missing_keypoint_low_confidence():
    kps = {"nose": Keypoint(10, 10, 0.1)}
    pose = Pose(person_id=1, keypoints=kps)
    assert not pose.is_visible("nose", 0.5)


def test_missing_keypoints_list(standing_pose):
    standing_pose.keypoints["left_wrist"].confidence = 0.1
    missing = standing_pose.missing_keypoints(0.4, required=["left_wrist", "right_wrist"])
    assert "left_wrist" in missing
    assert "right_wrist" not in missing


def test_visible_ratio_all_visible(standing_pose):
    assert standing_pose.visible_ratio(0.5) == 1.0


def test_from_array_builds_all_keypoints():
    arr = [[float(i), float(i) * 2, 0.9] for i in range(17)]
    pose = Pose.from_array(person_id=5, kp_array=arr)
    assert len(pose.keypoints) == len(COCO_KEYPOINTS)
    assert pose.person_id == 5
