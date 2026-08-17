import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from app.pose.keypoints import Pose, Keypoint, COCO_KEYPOINTS


def make_pose(person_id=1, overrides=None, conf=0.9, bbox=(0, 0, 100, 200), ts=0.0, frame_index=0):
    """
    Build a Pose representing a person STANDING upright, centered at
    x=100 in a 200-unit-tall body, facing the camera. `overrides` maps
    keypoint name -> (x, y, confidence) to move specific joints.
    """
    base = {
        "nose": (100, 20, conf),
        "left_eye": (97, 18, conf), "right_eye": (103, 18, conf),
        "left_ear": (94, 19, conf), "right_ear": (106, 19, conf),
        "left_shoulder": (85, 50, conf), "right_shoulder": (115, 50, conf),
        "left_elbow": (80, 90, conf), "right_elbow": (120, 90, conf),
        "left_wrist": (78, 130, conf), "right_wrist": (122, 130, conf),
        "left_hip": (90, 130, conf), "right_hip": (110, 130, conf),
        "left_knee": (90, 175, conf), "right_knee": (110, 175, conf),
        "left_ankle": (90, 220, conf), "right_ankle": (110, 220, conf),
    }
    if overrides:
        for k, v in overrides.items():
            base[k] = v
    kps = {name: Keypoint(x, y, c) for name, (x, y, c) in base.items()}
    return Pose(person_id=person_id, keypoints=kps, bbox=bbox, detection_confidence=0.9,
                frame_index=frame_index, timestamp=ts)


@pytest.fixture
def standing_pose():
    return make_pose()


@pytest.fixture
def sitting_pose():
    # Thigh roughly horizontal (hip->knee), shin roughly vertical (knee->ankle)
    # -> ~90 degree knee bend, the geometric signature of a seated posture.
    return make_pose(overrides={
        "left_knee": (120, 130, 0.9), "left_ankle": (120, 170, 0.9),
        "right_knee": (80, 130, 0.9), "right_ankle": (80, 170, 0.9),
    })


@pytest.fixture
def squat_down_pose():
    # Deep knee bend, torso still upright -> squat "down" position.
    return make_pose(overrides={
        "left_knee": (95, 165, 0.9), "left_ankle": (90, 175, 0.9),
        "right_knee": (105, 165, 0.9), "right_ankle": (110, 175, 0.9),
        "left_hip": (90, 165, 0.9), "right_hip": (110, 165, 0.9),
    })


@pytest.fixture
def bending_pose():
    # Torso tilted far from vertical, legs kept fairly straight.
    return make_pose(overrides={
        "left_shoulder": (40, 90, 0.9), "right_shoulder": (70, 90, 0.9),
        "left_elbow": (30, 120, 0.9), "right_elbow": (60, 120, 0.9),
        "left_wrist": (20, 150, 0.9), "right_wrist": (50, 150, 0.9),
        "nose": (30, 80, 0.9),
    })


@pytest.fixture
def lying_pose():
    # Horizontal body: shoulders and hips at nearly the same height, wide bbox.
    return make_pose(
        overrides={
            "left_shoulder": (40, 130, 0.9), "right_shoulder": (40, 110, 0.9),
            "left_elbow": (60, 130, 0.9), "right_elbow": (60, 110, 0.9),
            "left_wrist": (80, 130, 0.9), "right_wrist": (80, 110, 0.9),
            "left_hip": (120, 130, 0.9), "right_hip": (120, 110, 0.9),
            "left_knee": (160, 130, 0.9), "right_knee": (160, 110, 0.9),
            "left_ankle": (200, 130, 0.9), "right_ankle": (200, 110, 0.9),
            "nose": (20, 120, 0.9),
        },
        bbox=(0, 90, 220, 150),  # wide, short bbox
    )


@pytest.fixture
def hand_raised_pose():
    return make_pose(overrides={
        "left_wrist": (78, 10, 0.9),  # well above the shoulder (y=50)
    })
