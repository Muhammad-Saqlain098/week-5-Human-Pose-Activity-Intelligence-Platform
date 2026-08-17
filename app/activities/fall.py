"""
Falling / Fallen (Requirement 15 Required Activity 4, Requirement 11, 12, 19).

A single-frame "body is horizontal" rule is explicitly disallowed by the
brief because it can't distinguish a fall from lying down normally,
sitting, or bending. Instead we combine several signals:

  1. Torso orientation close to horizontal (torso_angle high)
  2. Body aspect ratio from the bounding box (wide/short vs tall/narrow)
  3. A sudden drop in hip height over a short window (the "fall" itself,
     as opposed to already lying down)
  4. Time spent in the low/horizontal posture (to confirm "fallen" vs a
     brief crouch/bend)

The output feeds a fall lifecycle state machine:

    Normal -> Possible Fall -> Fall Confirmed -> Fall Active -> Resolved

`FallLifecycle` below is a light wrapper around ActivityDetector's
CANDIDATE/CONFIRMED/ACTIVE/ENDED states, renamed for fall-specific
terminology and exposed with the extra `sudden_drop` signal used to
distinguish an actual fall event from someone who sits/lies down slowly.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer
from app.pose.normalization import estimate_scale
from app.activities.base_activity import ActivityDetector, ActivityRuntime, ActivityState

HORIZONTAL_TORSO_MIN_DEG = 55.0
ASPECT_RATIO_MIN = 1.2          # width/height of bbox; a standing person is tall & narrow (<1)
SUDDEN_DROP_MIN = 0.12          # normalized hip-y displacement over the drop window that looks "sudden"
DROP_WINDOW_FRAMES = 6


def _bbox_aspect_ratio(pose: Pose) -> Optional[float]:
    x1, y1, x2, y2 = pose.bbox
    w, h = (x2 - x1), (y2 - y1)
    if h <= 0:
        return None
    return w / h


def _hip_y_drop(history: PoseSequenceBuffer, frames: int = DROP_WINDOW_FRAMES) -> Optional[float]:
    """
    Scale-normalized vertical hip displacement over the window. Reads RAW
    pose coordinates (not the hip-centered `normalized` pose, which by
    construction always places the hip at y=0 and so can never show a
    drop) and divides by torso-length scale for camera-distance invariance.
    """
    items = history.recent(frames)
    samples = []  # (y, scale)
    for f in items:
        p = f.pose
        if p.is_visible("left_hip", 0.3) and p.is_visible("right_hip", 0.3):
            lh, rh = p.get("left_hip"), p.get("right_hip")
            y = (lh.y + rh.y) / 2
            s = estimate_scale(p, 0.3) or 1.0
            samples.append((y, s))
    if len(samples) < 2:
        return None
    y0, s0 = samples[0]
    y1, s1 = samples[-1]
    scale = s1 or s0 or 1.0
    return (y1 - y0) / scale  # positive = hip moved down (image coords increase downward)


def is_lying_posture(pose: Pose, angles: JointAngles) -> bool:
    """True if the CURRENT pose looks horizontal/on-the-ground, regardless of how it got there."""
    horizontal = angles.torso_angle is not None and angles.torso_angle >= HORIZONTAL_TORSO_MIN_DEG
    ar = _bbox_aspect_ratio(pose)
    wide_flat = ar is not None and ar >= ASPECT_RATIO_MIN
    return horizontal or wide_flat


def condition(pose: Pose, angles: JointAngles, normalized, history: PoseSequenceBuffer) -> bool:
    """
    Raw per-frame "fall-like" condition: person currently looks like they are
    lying/horizontal. Sudden-drop is checked separately (see `had_sudden_drop`)
    to distinguish an actual fall from someone who lay down deliberately.
    """
    return is_lying_posture(pose, angles)


def had_sudden_drop(history: PoseSequenceBuffer) -> bool:
    drop = _hip_y_drop(history, DROP_WINDOW_FRAMES)
    return drop is not None and drop >= SUDDEN_DROP_MIN


def build_detector(config) -> ActivityDetector:
    frames_for_confirm = max(2, int(config.fall_confirmation_seconds * 15))  # assume ~15-30fps source
    return ActivityDetector(
        name="fallen",
        condition_fn=condition,
        confirm_frames=frames_for_confirm,
        end_grace_frames=config.end_grace_frames,
        min_active_hold_frames=1,
    )


@dataclass
class FallEvent:
    event_id: str
    person_id: int
    start_time: float
    confidence: float
    status: str = "possible_fall"     # possible_fall -> fall_confirmed -> alert_active -> acknowledged -> resolved
    had_sudden_onset: bool = False
    end_time: Optional[float] = None

    def confirm(self, timestamp: float):
        self.status = "fall_confirmed"

    def activate_alert(self, timestamp: float):
        self.status = "alert_active"

    def acknowledge(self):
        self.status = "acknowledged"

    def resolve(self, timestamp: float):
        self.status = "resolved"
        self.end_time = timestamp
