# Activity State & Person State Documentation

## 1. Per-activity temporal state machine

Implemented once in `app/activities/base_activity.py` (`ActivityDetector`,
`ActivityRuntime`, `ActivityState`) and reused by all seven activity
rules, so every activity gets identical, well-tested smoothing behavior
(`tests/test_state.py`).

```
        condition True
IDLE ------------------------> CANDIDATE
  ^                                 |
  | condition False                 | condition True held for
  |                                 | confirm_frames consecutive frames
  |                                 v
  +---------------------------- CONFIRMED
                                     |
                          condition True next frame
                                     v
                                  ACTIVE  <---+
                                     |        | condition False,
                          condition False     | false_streak < end_grace_frames
                          false_streak >=     | (brief flicker tolerated)
                          end_grace_frames    |
                                     v        |
                                  ENDED -------+
```

- **CANDIDATE:** condition just became true; not yet trusted. Reverts
  straight to IDLE on a single false frame (prevents one noisy frame
  from starting an activity).
- **CONFIRMED:** condition held for `confirm_frames` consecutive frames
  (default 5, tunable per activity -- e.g. walking uses `confirm_frames - 2`
  since motion evidence is already multi-frame by construction).
- **ACTIVE:** the "reported" state once confirmed and still holding.
  Tolerates up to `end_grace_frames - 1` consecutive false frames without
  ending, so a single missed-keypoint frame doesn't end an activity.
- **ENDED:** condition has been false for `end_grace_frames` consecutive
  frames, or a CONFIRMED state saw a false frame immediately (fast-path
  end for activities that never reached ACTIVE).

`ActivityDetector.is_effectively_active()` treats CONFIRMED and ACTIVE
as "this activity is currently true" for the purposes of the
`ActivityManager`'s priority resolution.

## 2. Person state (per tracked ID)

`app/events/activity_manager.py:PersonState` holds, per Requirement 38:

| Field | Description |
|---|---|
| `person_id` | Track ID from the pose estimator / tracker |
| `pose_history` | `PoseSequenceBuffer` (rolling window, default 30 frames) |
| `runtimes` | `Dict[activity_name, ActivityRuntime]` -- one state machine per activity |
| `current_activity` / `previous_activity` | Resolved single activity label, and what it was before the last transition |
| `activity_start_time` | When the current activity began |
| `active_alerts` | List of currently-active alert type strings |
| `last_seen` / `first_seen` | Timestamps for tracking-expiry and lifetime |
| `timeline` | List of `TimelineEntry(activity, start_time, end_time, confidence)` -- the full activity history for this person |
| `squat_counter` | `SquatCounter` instance |
| `unsafe_bend_start` | Timestamp the current bending streak began (for the ergonomic warning) |
| `fall_event` | Current `FallEvent` lifecycle object, or `None` |
| `current_event_db_id` | Row ID of the currently-open `activity_events` record (so it can be closed with an end time later) |

### When state is created
On the first frame a track ID is seen (`ActivityManager._get_or_create_person`),
which also upserts a `persons` row in the database.

### How it is updated
Every frame, `ActivityManager.process_person()`:
1. extracts joint angles + normalized pose,
2. appends to the pose sequence buffer,
3. evaluates all configured activity rules and steps each one's state machine,
4. resolves a single reported activity by priority (see `docs/activity_rules.md`),
5. on a transition, closes the previous timeline/DB entry and opens a new one,
6. updates the squat counter, unsafe-bending timer, and fall lifecycle,
7. fires alerts (cooldown-gated) as needed.

### How it expires
`ActivityManager.expire_stale_people(current_time)` removes any person
whose `last_seen` is older than `track_expiry_seconds` (default 3s),
closing their open timeline/DB entry first so no activity event is left
dangling. Verified in `tests/test_integration.py::test_tracking_expiry_removes_stale_person`.

### Short tracking loss
Because `track_expiry_seconds` (default 3s) is deliberately longer than
a single missed detection, a person who is briefly occluded and
re-detected within that window keeps the *same* `PersonState` (assuming
the tracker successfully re-associates the ID) -- their pose history and
current activity persist across the gap rather than resetting.

### ID switching
If the underlying tracker assigns a *new* ID to the same physical person
(a tracker failure mode, not something this layer can detect on its
own), `ActivityManager` treats it as a brand-new person: a fresh
`PersonState` is created, and the old one eventually expires via
`track_expiry_seconds`. This is a known limitation -- mitigating it would
require re-identification (e.g. appearance embedding matching) which is
out of scope for Week 5.

## 3. Fall lifecycle (separate from, but driven by, the "fallen" activity state machine)

```
Normal --(fall rule -> CANDIDATE)--> Possible Fall
Possible Fall --(fall rule -> CONFIRMED/ACTIVE)--> Fall Confirmed --(alert fired)--> Alert Active
Alert Active --(operator action, optional)--> Acknowledged
Alert Active / Acknowledged --(fall rule -> ENDED, person recovers)--> Resolved
```

Implemented in `app/activities/fall.py:FallEvent` and driven by
`ActivityManager._handle_fall_lifecycle`, which watches the `fallen`
activity's `ActivityRuntime.state` and advances/creates/resolves the
`FallEvent` accordingly (`tests/test_fall.py::test_fall_lifecycle_confirm_and_resolve`).
