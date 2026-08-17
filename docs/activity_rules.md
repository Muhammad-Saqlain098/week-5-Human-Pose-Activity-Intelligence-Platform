# Activity Rule Specification

Every rule below is implemented as a per-frame boolean `condition()` in
`app/activities/<name>.py`, then smoothed by the shared temporal state
machine in `app/activities/base_activity.py` (see
`docs/state_management.md`). "Required keypoints" lists what must be
visible (confidence ≥ `keypoint_conf_threshold`, default 0.4) for the
rule to evaluate at all; if they're missing the condition simply
returns `False` rather than raising.

---

## Standing (Required Activity 1)
- **Purpose:** detect an upright, largely stationary person.
- **Required keypoints:** shoulders, hips, knees.
- **Joint angles used:** torso angle from vertical, knee angles.
- **Spatial conditions:** torso angle ≤ 20°; both/either knee angle ≥ 150° (leg fairly straight).
- **Temporal conditions:** hip motion over the last 10 frames < 0.03 (scale-normalized) — rules out walking.
- **Start / end condition:** state machine CANDIDATE after 1 true frame, CONFIRMED after `candidate_to_confirmed_frames` (default 5) consecutive true frames, ENDED after `end_grace_frames` (default 6) consecutive false frames.
- **Minimum duration to confirm:** 5 frames (~0.15-0.3s at 15-30fps).
- **Alert requirement:** none.
- **Known limitations:** a person standing very still with legs bent slightly (e.g. knees locked but camera angle foreshortens the leg) can be missed; requires visible knees, so it fails under leg occlusion (falls through to "no activity" rather than misclassifying).

## Sitting (Required Activity 2)
- **Purpose:** detect a seated posture.
- **Required keypoints:** hips, knees, shoulders.
- **Joint angles used:** average knee angle, torso angle.
- **Spatial conditions:** average knee angle in [70°, 130°]; torso angle ≤ 35° (still reasonably upright); normalized hip-y - shoulder-y ≥ 0.3 (hip sits well below shoulder, ruling out a standing person who merely leans).
- **Temporal conditions:** persistence via the shared state machine.
- **Start / end condition:** same CANDIDATE/CONFIRMED/ACTIVE/ENDED machine, 5-frame confirm.
- **Alert requirement:** none.
- **Known limitations (measured, see `docs/experiments.md`):** the geometric signature of sitting (bent knees + upright torso) overlaps with squatting; in our 32-scenario evaluation, sitting scenarios were resolved as "squatting" because squatting is checked first in the priority order and its knee-angle range is broader (≤120° vs 70-130°). Recall for `sitting` was 0% in that run — a genuine, documented limitation, not a hidden bug. A production fix would add a hip-height-vs-support-surface heuristic or require an explicit "seated for N seconds without leg movement" signal to disambiguate from a held squat.

## Walking (Required Activity 3)
- **Purpose:** detect locomotion through the scene.
- **Required keypoints:** hips, ankles, shoulders (torso angle).
- **Joint angles used:** torso angle (upright check only).
- **Spatial conditions:** torso angle ≤ 30°.
- **Temporal conditions:** requires ≥ 5 frames of history; scale-normalized hip displacement over the last 8 frames ≥ 0.02; at least one ankle's velocity ≥ 0.01 (rules out torso-sway-only motion).
- **Start / end condition:** confirm frames = `candidate_to_confirmed_frames - 2` (faster to confirm than static poses, since walking is inherently defined by motion already accumulated in the buffer).
- **Alert requirement:** none.
- **Known limitations:** cannot distinguish walking from being carried/pushed; a stationary person swaying (e.g. dancing in place) can under some jitter levels register motion above threshold — see "fast_movement" and "occlusion" difficulty results in `docs/experiments.md`.

## Falling / Fallen (Required Activity 4)
- **Purpose:** detect a fall event and the resulting "on the ground" state, distinct from sitting/bending/lying deliberately.
- **Required keypoints:** shoulders, hips (torso angle); bounding box (aspect ratio).
- **Joint angles used:** torso angle from vertical.
- **Spatial conditions:** torso angle ≥ 55° (near-horizontal) **or** bounding-box aspect ratio (width/height) ≥ 1.2 (wide-and-short silhouette).
- **Temporal conditions:** the raw per-frame "lying-like" condition must hold for `fall_confirmation_seconds` (default 0.6s, converted to frames assuming ~15-30fps) to reach CONFIRMED. Separately, `had_sudden_drop()` checks scale-normalized hip-y displacement over the last 6 frames ≥ 0.12 to flag a *sudden* onset (distinguishing an actual fall from lying down deliberately) — this is recorded on the `FallEvent` but does not gate confirmation, since a slow collapse is still a fall worth confirming.
- **Start / end condition:** `Normal → Possible Fall (CANDIDATE) → Fall Confirmed (CONFIRMED/ACTIVE) → Resolved (ENDED)`. See `docs/state_management.md` and `app/activities/fall.py:FallEvent`.
- **Minimum duration:** `fall_confirmation_seconds` (configurable, default 0.6s).
- **Alert requirement:** yes — `fall_detected`, high severity, cooldown-gated (`alert_cooldown_seconds`, default 15s), with an evidence frame saved via `EvidenceStore`.
- **Known limitations:** a single-frame horizontal check is explicitly avoided; however lying down normally (e.g. resting) and a genuine fall look geometrically identical once the person is down — only the sudden-onset flag (motion just before the horizontal pose) distinguishes them, and that heuristic can miss slow collapses or false-flag a quick sit-to-lie transition.

## Hand Raised (Required Activity 5)
- **Purpose:** detect one or both hands raised above the shoulder.
- **Required keypoints:** wrist + shoulder (either side).
- **Spatial conditions:** normalized (shoulder.y − wrist.y) > 0.15 (wrist above shoulder by a scale-relative margin) for the left or right arm.
- **Temporal conditions:** confirm after `hand_raise_min_frames` (default 5) consecutive true frames; ends quickly (3-frame grace) since hand-raise is usually a deliberate, short gesture.
- **Alert requirement:** none by default (can be wired to the alert engine if used as a "help" gesture).
- **Known limitations:** a raised elbow with a lowered wrist (unusual arm pose) is not counted as "raised"; deliberately conservative to avoid false positives from arm-swinging while walking.

## Bending (Additional Activity)
- **Purpose:** detect forward/downward torso bend while standing (not sitting/squatting).
- **Required keypoints:** shoulders, hips, knees, torso angle.
- **Spatial conditions:** torso angle ≥ 35°; knee angle ≥ 120° (legs not deeply bent, ruling out squat/sit).
- **Alert requirement:** none directly; feeds the separate **Unsafe Bending** ergonomic monitor in `ActivityManager._check_unsafe_bending`, which fires a `unsafe_bending` alert (medium severity) if torso angle stays ≥ `unsafe_bend_torso_angle` (default 45°) for ≥ `unsafe_bend_duration_seconds` (default 8s).
- **Known limitations:** this is explicitly documented (per Requirement 22) as a **prototype posture-risk indicator, not a certified ergonomic or medical assessment**.

## Squatting (Additional Activity, also the repetition-counted activity)
- **Purpose:** detect a deep knee bend with the torso still reasonably upright.
- **Required keypoints:** hips, knees, torso angle.
- **Spatial conditions:** max knee angle ≤ 120°; torso angle ≤ 45°.
- **Repetition counting:** a separate `SquatCounter` state machine (`app/activities/squatting.py`) tracks `standing → down → standing` and only increments the count when the person returns **fully** to standing (`squat_up_knee_angle`, default 160°), which prevents double-counting partial movements (verified in `tests/test_counter.py::test_squat_counter_does_not_double_count_partial_movement`).
- **Known limitations:** overlaps geometrically with sitting (see Sitting section above); in our priority order, squatting is checked before sitting, which is why sitting scenarios were mis-resolved as squatting in evaluation (measured 28.6% precision / 100% recall for squatting — over-triggering onto sitting frames).

---

## Priority resolution when multiple rules are true

`ACTIVITY_PRIORITY = [fallen, hand_raised, squatting, bending, sitting, walking, standing]`
in `app/events/activity_manager.py`. Safety-critical / more specific
activities win: a fall always overrides everything else; a raised hand
overrides posture activities; squatting/bending (dynamic, deliberate
postures) are checked before the more passive sitting/walking/standing.
This ordering is a documented design choice, and the evaluation results
above show its direct trade-off (sitting recall suffers because
squatting is checked first).
