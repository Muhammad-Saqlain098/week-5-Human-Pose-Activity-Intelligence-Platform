"""
Keypoint schema and Pose data structure (Requirement 2).

Uses the standard 17-point COCO keypoint layout, which is what
Ultralytics YOLO-Pose, MoveNet and most COCO-trained pose models output.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List

COCO_KEYPOINTS: List[str] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]
KEYPOINT_INDEX: Dict[str, int] = {name: i for i, name in enumerate(COCO_KEYPOINTS)}

# Pairs of keypoints that form the visual skeleton (for drawing / limb reasoning)
SKELETON_EDGES: List[Tuple[str, str]] = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"), ("right_knee", "right_ankle"),
    ("nose", "left_shoulder"), ("nose", "right_shoulder"),
]


@dataclass
class Keypoint:
    x: float
    y: float
    confidence: float = 0.0

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Pose:
    """A single person's pose in a single frame."""
    person_id: Optional[int]
    keypoints: Dict[str, Keypoint]
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # x1,y1,x2,y2
    detection_confidence: float = 0.0
    frame_index: int = 0
    timestamp: float = 0.0

    def get(self, name: str) -> Optional[Keypoint]:
        return self.keypoints.get(name)

    def is_visible(self, name: str, min_conf: float) -> bool:
        kp = self.keypoints.get(name)
        return kp is not None and kp.confidence >= min_conf

    def missing_keypoints(self, min_conf: float, required: Optional[List[str]] = None) -> List[str]:
        names = required or COCO_KEYPOINTS
        return [n for n in names if not self.is_visible(n, min_conf)]

    def visible_ratio(self, min_conf: float) -> float:
        if not self.keypoints:
            return 0.0
        visible = sum(1 for kp in self.keypoints.values() if kp.confidence >= min_conf)
        return visible / len(self.keypoints)

    @staticmethod
    def from_array(person_id: Optional[int], kp_array, bbox=(0, 0, 0, 0),
                    detection_confidence: float = 0.0, frame_index: int = 0,
                    timestamp: float = 0.0) -> "Pose":
        """
        Build a Pose from a raw (17,3) array/list of [x, y, confidence],
        e.g. the format returned by ultralytics `results.keypoints.data`.
        """
        kps: Dict[str, Keypoint] = {}
        for i, name in enumerate(COCO_KEYPOINTS):
            if i < len(kp_array):
                x, y, c = kp_array[i][0], kp_array[i][1], kp_array[i][2]
                kps[name] = Keypoint(float(x), float(y), float(c))
        return Pose(
            person_id=person_id, keypoints=kps, bbox=tuple(bbox),
            detection_confidence=detection_confidence,
            frame_index=frame_index, timestamp=timestamp,
        )
