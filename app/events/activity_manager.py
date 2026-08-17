"""
Activity Manager (Requirement 4, 15, 19, 21, 23, Stage 6-11).

This is the orchestration layer: for every tracked person it maintains
pose history, evaluates every configured activity detector, resolves
which single activity should be reported when several rules are true at
once (priority order), advances the fall lifecycle, drives the squat
counter and the unsafe-bending ergonomic warning, and writes
transitions to the activity timeline / database / alert engine.
"""
from __future__ import annotations
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from app.pose.keypoints import Pose
from app.pose.angles import compute_joint_angles, JointAngles
from app.pose.normalization import normalize_pose
from app.pose.sequence import PoseSequenceBuffer
from app.activities.base_activity import ActivityDetector, ActivityRuntime, ActivityState
from app.activities import standing, sitting, walking, hand_raise, bending, squatting, fall
from app.events.alerts import AlertEngine
from app.events.evidence import EvidenceStore

# Priority order when multiple rule conditions are simultaneously true.
# Safety-critical / more specific activities win over generic ones.
ACTIVITY_PRIORITY = [
    "fallen", "hand_raised", "squatting", "bending", "sitting", "walking", "standing",
]


@dataclass
class TimelineEntry:
    activity: str
    start_time: float
    end_time: Optional[float] = None
    confidence: float = 0.0

    def duration(self) -> Optional[float]:
        if self.end_time is None:
            return None
        return self.end_time - self.start_time


@dataclass
class PersonState:
    person_id: int
    pose_history: PoseSequenceBuffer
    runtimes: Dict[str, ActivityRuntime] = field(default_factory=dict)
    current_activity: Optional[str] = None
    previous_activity: Optional[str] = None
    activity_start_time: Optional[float] = None
    active_alerts: List[str] = field(default_factory=list)
    last_seen: float = 0.0
    first_seen: float = 0.0
    timeline: List[TimelineEntry] = field(default_factory=list)
    squat_counter: Optional[squatting.SquatCounter] = None
    unsafe_bend_start: Optional[float] = None
    fall_event: Optional[fall.FallEvent] = None
    current_event_db_id: Optional[int] = None
    last_angles: Optional[JointAngles] = None
    motion: float = 0.0


class ActivityManager:
    def __init__(self, config, db=None, source_id: str = "default"):
        self.config = config
        self.db = db
        self.source_id = source_id
        self.alert_engine = AlertEngine(cooldown_seconds=config.alert_cooldown_seconds)
        self.evidence = EvidenceStore(config.evidence_dir)
        self.people: Dict[int, PersonState] = {}

        self.detectors: Dict[str, ActivityDetector] = {
            "standing": standing.build_detector(config),
            "sitting": sitting.build_detector(config),
            "walking": walking.build_detector(config),
            "hand_raised": hand_raise.build_detector(config),
            "bending": bending.build_detector(config),
            "squatting": squatting.build_detector(config),
            "fallen": fall.build_detector(config),
        }
        # only keep configured/selected activities active
        self.active_activity_names = [a for a in ACTIVITY_PRIORITY if a in config.selected_activities]

    # ---------------- person lifecycle ----------------

    def _get_or_create_person(self, person_id: int, timestamp: float) -> PersonState:
        if person_id not in self.people:
            self.people[person_id] = PersonState(
                person_id=person_id,
                pose_history=PoseSequenceBuffer(maxlen=self.config.sequence_length),
                runtimes={name: ActivityRuntime() for name in self.detectors},
                first_seen=timestamp,
                last_seen=timestamp,
                squat_counter=squatting.SquatCounter(self.config),
            )
            if self.db:
                self.db.upsert_person(person_id, timestamp)
        return self.people[person_id]

    def expire_stale_people(self, current_time: float) -> List[int]:
        expired = [
            pid for pid, p in self.people.items()
            if (current_time - p.last_seen) > self.config.track_expiry_seconds
        ]
        for pid in expired:
            self._close_current_activity(self.people[pid], current_time)
            del self.people[pid]
        return expired

    # ---------------- core per-frame processing ----------------

    def process_person(self, pose: Pose, frame=None, timestamp: Optional[float] = None) -> PersonState:
        """
        Feed one person's pose for the current frame through the full
        pipeline: feature extraction -> sequence buffer -> activity rules
        -> temporal smoothing -> resolution -> events/alerts/db.
        """
        ts = timestamp if timestamp is not None else time.time()
        person_id = pose.person_id if pose.person_id is not None else -1
        state = self._get_or_create_person(person_id, ts)
        state.last_seen = ts
        if self.db:
            self.db.upsert_person(person_id, ts)

        # --- feature extraction (Requirement 6, 7) ---
        angles = compute_joint_angles(pose, self.config.keypoint_conf_threshold)
        normalized = normalize_pose(pose, self.config.keypoint_conf_threshold)
        state.pose_history.add(pose, angles, normalized, ts)
        state.last_angles = angles
        motion = state.pose_history.hip_motion(frames=10)
        state.motion = motion if motion is not None else 0.0

        # --- evaluate every configured rule (Requirement 9, 17) ---
        confirmed_candidates: List[str] = []
        for name in self.active_activity_names:
            detector = self.detectors[name]
            cond = detector.evaluate_condition(pose, angles, normalized, state.pose_history)
            runtime = detector.update(state.runtimes[name], cond, ts)
            state.runtimes[name] = runtime
            if detector.is_effectively_active(runtime):
                confirmed_candidates.append(name)

        # --- resolve to a single reported activity by priority (Requirement 10) ---
        resolved = None
        for name in ACTIVITY_PRIORITY:
            if name in confirmed_candidates:
                resolved = name
                break

        self._apply_transition(state, resolved, ts)

        # --- squat repetition counting (Requirement 13) ---
        if state.squat_counter and angles is not None:
            state.squat_counter.update(angles)

        # --- unsafe bending ergonomic monitor (Requirement 14, 22) ---
        self._check_unsafe_bending(state, angles, ts, frame)

        # --- fall lifecycle & alerting (Requirement 11, 12, 19, 20, 26) ---
        self._handle_fall_lifecycle(state, resolved, angles, ts, frame)

        return state

    # ---------------- transition / timeline handling ----------------

    def _apply_transition(self, state: PersonState, resolved: Optional[str], ts: float):
        if resolved == state.current_activity:
            return  # no change -- avoid noisy per-frame relabeling

        self._close_current_activity(state, ts)

        state.previous_activity = state.current_activity
        state.current_activity = resolved
        state.activity_start_time = ts if resolved else None

        if resolved:
            entry = TimelineEntry(activity=resolved, start_time=ts)
            state.timeline.append(entry)
            confidence = state.runtimes[resolved].true_streak / max(1, self.detectors[resolved].confirm_frames)
            confidence = min(1.0, confidence)
            if self.db:
                state.current_event_db_id = self.db.insert_activity_event(
                    person_id=state.person_id, activity=resolved, start_time=ts,
                    confidence=confidence, source_id=self.source_id,
                )

    def _close_current_activity(self, state: PersonState, ts: float):
        if state.current_activity and state.timeline:
            state.timeline[-1].end_time = ts
        if self.db and state.current_event_db_id is not None:
            self.db.close_activity_event(state.current_event_db_id, ts)
        state.current_event_db_id = None

    # ---------------- unsafe bending ----------------

    def _check_unsafe_bending(self, state: PersonState, angles: JointAngles, ts: float, frame):
        is_bent = angles.torso_angle is not None and angles.torso_angle >= self.config.unsafe_bend_torso_angle
        if is_bent:
            if state.unsafe_bend_start is None:
                state.unsafe_bend_start = ts
            elif (ts - state.unsafe_bend_start) >= self.config.unsafe_bend_duration_seconds:
                evidence_path = self.evidence.save_frame(frame, "bending", state.person_id, "unsafe_bend")
                alert = self.alert_engine.fire(
                    person_id=state.person_id, alert_type="unsafe_bending", timestamp=ts,
                    message=f"Person {state.person_id} bending posture exceeded "
                            f"{self.config.unsafe_bend_duration_seconds}s (prototype ergonomic warning, "
                            f"not a certified assessment).",
                    severity="medium", evidence_path=evidence_path,
                )
                if alert:
                    state.active_alerts.append("unsafe_bending")
                    if self.db:
                        self.db.insert_alert(state.person_id, "unsafe_bending", ts, alert.message)
        else:
            state.unsafe_bend_start = None
            if "unsafe_bending" in state.active_alerts:
                state.active_alerts.remove("unsafe_bending")

    # ---------------- fall lifecycle ----------------

    def _handle_fall_lifecycle(self, state: PersonState, resolved: Optional[str],
                                angles: JointAngles, ts: float, frame):
        runtime = state.runtimes["fallen"]

        if runtime.state in (ActivityState.CANDIDATE,) and state.fall_event is None:
            state.fall_event = fall.FallEvent(
                event_id=str(uuid.uuid4())[:8], person_id=state.person_id,
                start_time=runtime.start_time or ts, confidence=0.3,
            )
            state.fall_event.had_sudden_onset = fall.had_sudden_drop(state.pose_history)

        if runtime.state in (ActivityState.CONFIRMED, ActivityState.ACTIVE) and state.fall_event:
            if state.fall_event.status == "possible_fall":
                state.fall_event.confirm(ts)
                evidence_path = self.evidence.save_frame(frame, "fall", state.person_id, "confirmed")
                alert = self.alert_engine.fire(
                    person_id=state.person_id, alert_type="fall_detected", timestamp=ts,
                    message=f"Fall confirmed for person {state.person_id} "
                            f"(sudden onset={state.fall_event.had_sudden_onset}).",
                    severity="high", evidence_path=evidence_path,
                )
                if alert:
                    state.fall_event.activate_alert(ts)
                    state.active_alerts.append("fall_detected")
                    if self.db:
                        self.db.insert_alert(state.person_id, "fall_detected", ts, alert.message,
                                              event_id=state.current_event_db_id)

        if runtime.state == ActivityState.ENDED and state.fall_event and state.fall_event.status != "resolved":
            state.fall_event.resolve(ts)
            if "fall_detected" in state.active_alerts:
                state.active_alerts.remove("fall_detected")
            state.fall_event = None

    # ---------------- reporting ----------------

    def snapshot(self) -> List[Dict[str, Any]]:
        """A lightweight live snapshot for the dashboard / main loop overlay."""
        out = []
        for pid, state in self.people.items():
            out.append({
                "person_id": pid,
                "current_activity": state.current_activity,
                "previous_activity": state.previous_activity,
                "active_alerts": list(state.active_alerts),
                "squat_count": state.squat_counter.count if state.squat_counter else 0,
                "motion": round(state.motion, 4),
                "last_seen": state.last_seen,
            })
        return out
