"""
Base class for rule-based activity detectors, implementing the temporal
state machine required by Requirement 10:

    CANDIDATE -> CONFIRMED -> ACTIVE -> ENDED

A raw per-frame boolean condition (implemented by each activity subclass)
is smoothed over time so a single flickering frame can't create or destroy
an activity label. This directly satisfies the "do not change activity
labels on every frame" requirement and reduces false alerts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from app.pose.keypoints import Pose
from app.pose.angles import JointAngles
from app.pose.sequence import PoseSequenceBuffer


class ActivityState(Enum):
    IDLE = "idle"                # condition not currently met, no activity
    CANDIDATE = "candidate"      # condition just became true, not yet confirmed
    CONFIRMED = "confirmed"      # condition held long enough to be confirmed
    ACTIVE = "active"            # confirmed and ongoing
    ENDED = "ended"              # was active, condition stopped holding


@dataclass
class ActivityRuntime:
    """Per-person, per-activity runtime state."""
    state: ActivityState = ActivityState.IDLE
    true_streak: int = 0
    false_streak: int = 0
    start_time: Optional[float] = None
    confirmed_time: Optional[float] = None
    end_time: Optional[float] = None
    last_condition: bool = False


ConditionFn = Callable[[Pose, JointAngles, Optional[dict], PoseSequenceBuffer], bool]


class ActivityDetector:
    """
    Wraps a raw per-frame `condition_fn` with confirm/end frame-count
    smoothing. Each activity module (standing.py, sitting.py, ...)
    instantiates one of these rather than re-implementing state logic.
    """

    def __init__(
        self,
        name: str,
        condition_fn: ConditionFn,
        confirm_frames: int = 5,
        end_grace_frames: int = 6,
        min_active_hold_frames: int = 3,
    ):
        self.name = name
        self.condition_fn = condition_fn
        self.confirm_frames = confirm_frames
        self.end_grace_frames = end_grace_frames
        self.min_active_hold_frames = min_active_hold_frames

    def evaluate_condition(self, pose: Pose, angles: JointAngles, normalized: Optional[dict],
                            history: PoseSequenceBuffer) -> bool:
        return self.condition_fn(pose, angles, normalized, history)

    def update(self, runtime: ActivityRuntime, condition: bool, timestamp: float) -> ActivityRuntime:
        runtime.last_condition = condition

        if condition:
            runtime.false_streak = 0
            runtime.true_streak += 1
        else:
            runtime.true_streak = 0
            runtime.false_streak += 1

        if runtime.state == ActivityState.IDLE:
            if condition:
                runtime.state = ActivityState.CANDIDATE
                runtime.start_time = timestamp
                runtime.true_streak = 1

        elif runtime.state == ActivityState.CANDIDATE:
            if not condition:
                runtime.state = ActivityState.IDLE
                runtime.start_time = None
            elif runtime.true_streak >= self.confirm_frames:
                runtime.state = ActivityState.CONFIRMED
                runtime.confirmed_time = timestamp

        elif runtime.state == ActivityState.CONFIRMED:
            if condition:
                runtime.state = ActivityState.ACTIVE
            else:
                runtime.state = ActivityState.ENDED
                runtime.end_time = timestamp

        elif runtime.state == ActivityState.ACTIVE:
            if not condition:
                if runtime.false_streak >= self.end_grace_frames:
                    runtime.state = ActivityState.ENDED
                    runtime.end_time = timestamp
                # else: stay ACTIVE through brief flicker (grace period)

        elif runtime.state == ActivityState.ENDED:
            # reset to idle/candidate next frame
            if condition:
                runtime.state = ActivityState.CANDIDATE
                runtime.start_time = timestamp
                runtime.true_streak = 1
            else:
                runtime.state = ActivityState.IDLE
                runtime.start_time = None

        return runtime

    def is_effectively_active(self, runtime: ActivityRuntime) -> bool:
        return runtime.state in (ActivityState.CONFIRMED, ActivityState.ACTIVE)
