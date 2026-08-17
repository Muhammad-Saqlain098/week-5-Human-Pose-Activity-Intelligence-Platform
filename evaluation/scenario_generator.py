"""
Synthetic evaluation-scenario generator (Requirement 32).

IMPORTANT / HONEST LIMITATION (documented in docs/experiments.md and the
builder journal): this sandbox has no camera and cannot download a real
video dataset, so we cannot run YOLO-Pose on real footage here. Instead,
we generate synthetic, ground-truth-labelled pose *sequences* directly
in normalized keypoint space and feed them through the exact same
ActivityManager / activity-detector code path that a real video would
use. This evaluates the activity-recognition logic layer honestly and
reproducibly. On a real deployment, the ONLY thing that changes is the
source of Pose objects (YoloPoseEstimator.infer() instead of this
generator) -- the recognition pipeline underneath is identical.

Each scenario returns a list of (Pose, timestamp) tuples and a ground
truth label for what a human reviewer would call the activity.
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

from app.pose.keypoints import Pose, Keypoint


def _pose(overrides, bbox=(0, 0, 100, 200), conf=0.9, jitter=0.0, rng=None):
    base = {
        "nose": (100, 20), "left_eye": (97, 18), "right_eye": (103, 18),
        "left_ear": (94, 19), "right_ear": (106, 19),
        "left_shoulder": (85, 50), "right_shoulder": (115, 50),
        "left_elbow": (80, 90), "right_elbow": (120, 90),
        "left_wrist": (78, 130), "right_wrist": (122, 130),
        "left_hip": (90, 130), "right_hip": (110, 130),
        "left_knee": (90, 175), "right_knee": (110, 175),
        "left_ankle": (90, 220), "right_ankle": (110, 220),
    }
    base.update(overrides)
    kps = {}
    for name, (x, y) in base.items():
        if jitter and rng:
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        kps[name] = Keypoint(x, y, conf)
    return Pose(person_id=1, keypoints=kps, bbox=bbox, detection_confidence=0.9)


@dataclass
class Scenario:
    name: str
    label: str                 # ground-truth activity
    frames: List[Tuple[Pose, float]]
    difficulty: str = "normal"  # normal | occlusion | low_light | multi_person | partial | angle | fast_motion


def _standing_frames(n, fps, jitter, rng):
    out = []
    for i in range(n):
        p = _pose({}, jitter=jitter, rng=rng)
        out.append((p, i / fps))
    return out


def _sitting_frames(n, fps, jitter, rng):
    out = []
    for i in range(n):
        p = _pose({
            "left_knee": (120, 130), "left_ankle": (120, 170),
            "right_knee": (80, 130), "right_ankle": (80, 170),
        }, jitter=jitter, rng=rng)
        out.append((p, i / fps))
    return out


def _walking_frames(n, fps, jitter, rng):
    out = []
    for i in range(n):
        shift = i * 5
        sway = 6 * math.sin(i * 0.9)  # alternating stride
        p = _pose({
            "left_hip": (90 + shift, 130), "right_hip": (110 + shift, 130),
            "left_ankle": (90 + shift + sway, 220), "right_ankle": (110 + shift - sway, 220),
            "left_knee": (90 + shift, 175), "right_knee": (110 + shift, 175),
            "left_shoulder": (85 + shift, 50), "right_shoulder": (115 + shift, 50),
            "left_elbow": (80 + shift, 90), "right_elbow": (120 + shift, 90),
            "left_wrist": (78 + shift, 130), "right_wrist": (122 + shift, 130),
            "nose": (100 + shift, 20),
        }, bbox=(0 + shift, 0, 100 + shift, 200), jitter=jitter, rng=rng)
        out.append((p, i / fps))
    return out


def _hand_raised_frames(n, fps, jitter, rng):
    out = []
    for i in range(n):
        p = _pose({"left_wrist": (78, 5)}, jitter=jitter, rng=rng)
        out.append((p, i / fps))
    return out


def _bending_frames(n, fps, jitter, rng):
    out = []
    for i in range(n):
        p = _pose({
            "left_shoulder": (40, 90), "right_shoulder": (70, 90),
            "left_elbow": (30, 120), "right_elbow": (60, 120),
            "left_wrist": (20, 150), "right_wrist": (50, 150),
            "nose": (30, 80),
        }, jitter=jitter, rng=rng)
        out.append((p, i / fps))
    return out


def _squatting_frames(n, fps, jitter, rng):
    out = []
    for i in range(n):
        p = _pose({
            "left_knee": (95, 165), "left_ankle": (90, 175),
            "right_knee": (105, 165), "right_ankle": (110, 175),
            "left_hip": (90, 165), "right_hip": (110, 165),
        }, jitter=jitter, rng=rng)
        out.append((p, i / fps))
    return out


def _fallen_frames(n, fps, jitter, rng, sudden=True):
    out = []
    stand_frames = 4 if sudden else 0
    for i in range(stand_frames):
        out.append((_pose({}, jitter=jitter, rng=rng), i / fps))
    for i in range(n - stand_frames):
        p = _pose({
            "left_shoulder": (40, 130), "right_shoulder": (40, 110),
            "left_elbow": (60, 130), "right_elbow": (60, 110),
            "left_wrist": (80, 130), "right_wrist": (80, 110),
            "left_hip": (120, 130), "right_hip": (120, 110),
            "left_knee": (160, 130), "right_knee": (160, 110),
            "left_ankle": (200, 130), "right_ankle": (200, 110),
            "nose": (20, 120),
        }, bbox=(0, 90, 220, 150), jitter=jitter, rng=rng)
        out.append((p, (stand_frames + i) / fps))
    return out


def build_evaluation_set(seed: int = 42, fps: float = 15.0) -> List[Scenario]:
    """
    Builds the 30-scenario evaluation set required by Section 32:
    5 each of standing/sitting/walking/hand-raised/fallen, plus 5
    additional-activity scenarios (bending/squatting), including
    several "difficult" variants (heavier jitter simulating occlusion,
    low light noise, partial visibility, fast motion, camera angle).
    """
    rng = random.Random(seed)
    scenarios: List[Scenario] = []

    def add_set(prefix, label, frame_fn, count=5, n_frames=25):
        difficulties = ["normal", "normal", "normal", "occlusion", "low_light"]
        for i in range(count):
            diff = difficulties[i % len(difficulties)]
            jitter = {"normal": 0.5, "occlusion": 4.0, "low_light": 2.5}.get(diff, 0.5)
            frames = frame_fn(n_frames, fps, jitter, rng)
            scenarios.append(Scenario(name=f"{prefix}_{i+1}", label=label, frames=frames, difficulty=diff))

    add_set("standing", "standing", _standing_frames)
    add_set("sitting", "sitting", _sitting_frames)
    add_set("walking", "walking", _walking_frames, n_frames=30)
    add_set("hand_raised", "hand_raised", _hand_raised_frames, n_frames=15)
    add_set("fallen", "fallen", lambda n, fps, j, r: _fallen_frames(n, fps, j, r, sudden=True), n_frames=20)

    # additional activities (5 total, split across bending/squatting)
    for i in range(3):
        frames = _bending_frames(25, fps, 0.5 if i < 2 else 3.0, rng)
        scenarios.append(Scenario(name=f"bending_{i+1}", label="bending", frames=frames,
                                   difficulty="normal" if i < 2 else "occlusion"))
    for i in range(2):
        frames = _squatting_frames(25, fps, 0.5, rng)
        scenarios.append(Scenario(name=f"squatting_{i+1}", label="squatting", frames=frames, difficulty="normal"))

    # extra difficult cases: fast movement walking, partial visibility sitting
    fast_walk = _walking_frames(15, fps * 2, 0.5, rng)  # compressed in time -> "fast movement"
    scenarios.append(Scenario(name="walking_fast", label="walking", frames=fast_walk, difficulty="fast_movement"))

    partial_pose_frames = []
    for i in range(20):
        p = _pose({}, jitter=0.5, rng=rng)
        # simulate partial visibility: zero out confidence on leg keypoints
        for name in ("left_ankle", "right_ankle", "left_knee", "right_knee"):
            p.keypoints[name].confidence = 0.05
        partial_pose_frames.append((p, i / fps))
    scenarios.append(Scenario(name="standing_partial_visibility", label="standing",
                               frames=partial_pose_frames, difficulty="partial_visibility"))

    return scenarios
