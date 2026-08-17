"""
Central configuration for the Human Pose and Activity Intelligence Platform.

Configuration is loaded from config.json if present, otherwise the defaults
below are used. Call Config.save() to persist changes back to disk so they
survive an application restart (Requirement 21).
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, asdict, field
from typing import Optional, List

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")


@dataclass
class Config:
    # --- Pose model (Section 7) ---
    pose_model: str = "yolov8n-pose.pt"      # ultralytics pose checkpoint
    detection_conf_threshold: float = 0.5     # min person-detection confidence
    keypoint_conf_threshold: float = 0.4      # min keypoint confidence to trust a joint

    # --- Sequence buffer (Requirement 5) ---
    sequence_length: int = 30                 # frames of pose history kept per person (15-60 recommended)

    # --- Temporal state machine (Requirement 10), frame counts @ ~30fps ---
    candidate_to_confirmed_frames: int = 5     # frames a condition must hold to confirm an activity
    confirmed_min_hold_frames: int = 3         # frames activity must remain true to stay "active"
    end_grace_frames: int = 6                  # frames condition may be false before activity "ends"

    # --- Fall detection (Requirement 11/12) ---
    fall_confirmation_seconds: float = 0.6     # time a fall-like pose must persist to confirm
    fall_lying_seconds: float = 1.0            # time lying down before "Fall Active"
    alert_cooldown_seconds: float = 15.0       # min gap between repeated alerts for same person/type

    # --- Hand raised ---
    hand_raise_min_frames: int = 5

    # --- Squat counting (Requirement 13) ---
    squat_down_knee_angle: float = 110.0       # knee angle (deg) below this = "down"
    squat_up_knee_angle: float = 160.0         # knee angle (deg) above this = "standing"

    # --- Unsafe bending / ergonomics (Requirement 14) ---
    unsafe_bend_torso_angle: float = 45.0      # degrees from vertical considered "bent"
    unsafe_bend_duration_seconds: float = 8.0

    # --- Walking ---
    walking_motion_threshold: float = 0.02     # normalized hip displacement/frame to count as walking

    # --- Tracking (Requirement 4) ---
    tracker: str = "bytetrack"                 # bytetrack | botsort | simple_iou
    track_expiry_seconds: float = 3.0          # drop a person's state after this long unseen

    # --- Video / app ---
    default_source: str = "0"                  # "0" = webcam index 0, else a path or RTSP url
    show_skeleton: bool = True
    frame_width: Optional[int] = None
    frame_height: Optional[int] = None

    # --- Storage ---
    db_path: str = "database/activity_events.db"
    evidence_dir: str = "evidence"

    # --- Selected activities (Requirement 21) ---
    selected_activities: List[str] = field(default_factory=lambda: [
        "standing", "sitting", "walking", "hand_raised",
        "fallen", "bending", "squatting",
    ])

    def save(self, path: str = CONFIG_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = CONFIG_PATH) -> "Config":
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            # ignore unknown keys so old config.json files don't break new fields
            valid = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
            return cls(**valid)
        cfg = cls()
        cfg.save(path)
        return cfg


if __name__ == "__main__":
    c = Config.load()
    print("Config initialized at", CONFIG_PATH)
