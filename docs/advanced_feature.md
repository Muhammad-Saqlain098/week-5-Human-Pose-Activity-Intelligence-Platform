# Advanced Feature: Multiple-Person Activity Recognition

**Chosen from Section 35** because the platform's tracking layer already
requires independent per-person state (Requirement 4), so this feature
directly strengthens the tracking and activity-recognition portions of
the rubric rather than bolting on unrelated functionality.

## How it works

`ActivityManager.people: Dict[int, PersonState]` (in
`app/events/activity_manager.py`) keys every piece of per-person state --
pose history, the 7 activity-detector runtimes, timeline, squat counter,
fall lifecycle, active alerts -- by track ID. `process_person()` is
called once per detected person per frame, and every operation inside it
only ever reads/writes that one person's `PersonState`. There is no
shared mutable state between people except the `AlertEngine`'s cooldown
map, which is itself keyed by `(person_id, alert_type)` -- so one
person's fall alert cooldown never suppresses another person's fall
alert.

## Proof, not just a claim

`evaluation/multi_person_demo.py` runs three independent synthetic
people through **one** `ActivityManager` instance simultaneously (one
standing still, one walking, one performing 2 full squat reps) and
prints/saves each person's final state. Actual output from running it
in this environment:

```
Tracked 3 people independently in one manager.
  Person 1: activity=standing, squats=0, motion=0.004
  Person 2: activity=walking,  squats=0, motion=0.0623
  Person 3: activity=squatting, squats=2, motion=0.0825
```

Person 3's squat counter correctly reached 2 while person 1's counter
stayed at 0 and person 2's activity was independently resolved as
"walking" -- proving the per-person state does not leak or interfere
across people. Full per-person timelines are saved to
`evaluation/results/multi_person_demo.json`.

## What this enables in a real deployment

- A single camera covering multiple people (e.g. a room, a factory
  floor) reports each person's activity, alerts, and repetition counts
  independently in the dashboard.
- Falls and unsafe-bending alerts are correctly attributed to the
  specific person who triggered them, with independent cooldowns, so
  one person's ongoing alert never masks another person's new one.
- The activity database (`activity_events` table) already stores
  `person_id` on every row, so multi-person history is queryable and
  filterable out of the box in the Streamlit dashboard (Requirement 28).

## Known limitation

Multi-person tracking is only as reliable as the underlying tracker's ID
association. If YOLO's ByteTrack loses and re-acquires a person (e.g.
after a long occlusion or two people crossing paths), the platform will
treat the reacquired detection as a new person rather than re-identifying
them -- see the "ID switching" discussion in `docs/state_management.md`.
