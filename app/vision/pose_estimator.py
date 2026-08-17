"""
Pose estimation wrapper around Ultralytics YOLO-Pose (Section 7).

Why YOLO-Pose for this project:
  - Single model does person detection + 17-keypoint (COCO) estimation
    in one forward pass, which simplifies the "Person Detection" +
    "Pose Estimation" stages of the required architecture into one call.
  - Ships with a built-in multi-object tracker (ByteTrack / BoT-SORT) via
    `model.track(..., persist=True)`, satisfying Requirement 4 without a
    separate tracking library.
  - Runs acceptably on CPU with the "n" (nano) checkpoint for a fellowship
    project; GPU is optional but recommended for higher FPS.

Keypoint format: standard COCO-17 (see app/pose/keypoints.py), in the
order [nose, eyes(2), ears(2), shoulders(2), elbows(2), wrists(2),
hips(2), knees(2), ankles(2)], each as [x, y, confidence].

Known limitations (documented per Section 7):
  - Struggles with heavy occlusion, extreme poses (e.g. lying flat,
    unusual camera angles), and very small/distant people.
  - Nano checkpoint trades accuracy for speed; use `yolov8s-pose.pt` /
    `yolov8m-pose.pt` for better accuracy at lower FPS.
  - Keypoint confidence can be inflated in low-light / motion blur.

Hardware: yolov8n-pose runs in real time (15-30+ FPS) on a modern CPU at
640px input; a GPU is recommended for larger checkpoints or higher
resolution/multi-camera setups.
"""
from __future__ import annotations
import logging
from typing import List, Optional

from app.pose.keypoints import Pose

logger = logging.getLogger("pose_estimator")

try:
    from ultralytics import YOLO
    _HAS_ULTRALYTICS = True
except ImportError:  # pragma: no cover
    _HAS_ULTRALYTICS = False


class YoloPoseEstimator:
    def __init__(self, model_path: str = "yolov8n-pose.pt", detection_conf: float = 0.5,
                 tracker: str = "bytetrack"):
        if not _HAS_ULTRALYTICS:
            raise RuntimeError(
                "ultralytics is not installed. Run: pip install ultralytics"
            )
        self.model = YOLO(model_path)
        self.detection_conf = detection_conf
        tracker_cfg = "bytetrack.yaml" if tracker == "bytetrack" else "botsort.yaml"
        self.tracker_cfg = tracker_cfg

    def infer(self, frame, frame_index: int = 0, timestamp: float = 0.0, use_tracking: bool = True) -> List[Pose]:
        """
        Run detection + pose estimation (+ tracking IDs) on a single frame.
        Returns a list of Pose objects, one per detected person.
        """
        if use_tracking:
            results = self.model.track(
                frame, conf=self.detection_conf, persist=True,
                tracker=self.tracker_cfg, verbose=False,
            )
        else:
            results = self.model.predict(frame, conf=self.detection_conf, verbose=False)

        poses: List[Pose] = []
        if not results:
            return poses
        result = results[0]
        if result.keypoints is None or result.boxes is None:
            return poses

        kp_data = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes = result.boxes
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
        confs = boxes.conf.cpu().numpy() if boxes.conf is not None else None
        xyxy = boxes.xyxy.cpu().numpy() if boxes.xyxy is not None else None

        for i in range(kp_data.shape[0]):
            person_id = int(ids[i]) if ids is not None else i
            det_conf = float(confs[i]) if confs is not None else 0.0
            bbox = tuple(xyxy[i]) if xyxy is not None else (0, 0, 0, 0)
            pose = Pose.from_array(
                person_id=person_id, kp_array=kp_data[i], bbox=bbox,
                detection_confidence=det_conf, frame_index=frame_index, timestamp=timestamp,
            )
            poses.append(pose)
        return poses
