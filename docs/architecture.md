# Architecture Documentation
Human Pose and Activity Intelligence Platform — Week 5

## 1. Overview

The platform turns "a person is present" into "what is the person
doing" by running every frame through a pipeline that moves from raw
pixels to a durable, queryable activity record:

```
Video Source (webcam / file / RTSP)
        |
        v
Person Detection + Pose Estimation   <-- single YOLO-Pose forward pass
        |
        v
Person Tracking (ByteTrack, ID-persistent)
        |
        v
Pose Sequence Buffer (per person, 15-60 frames)
        |
        v
Feature Extraction (joint angles, normalization, velocity)
        |
        v
Activity Recognition Engine (7 rule-based detectors + temporal state machine)
        |
        +----------------------------+
        |                            |
        v                            v
Event Manager                Activity Database (SQLite)
  - Timeline                        |
  - Squat counter                   v
  - Unsafe-bending monitor    Analytics Dashboard (Streamlit)
  - Fall lifecycle
        |
        v
Alert Engine (cooldown-gated fall / unsafe-bending alerts + evidence capture)
```

This matches the architecture required in Section 5 / Section 36 of the
brief, merging "Person Detection" and "Pose Estimation" into one stage
because YOLO-Pose performs both in a single model pass.

## 2. Component responsibilities

| Component | File(s) | Responsibility |
|---|---|---|
| Video Source | `app/vision/video_source.py` | Uniform interface over webcam index, video file path, or RTSP URL |
| Pose Estimator | `app/vision/pose_estimator.py` | Wraps Ultralytics YOLO-Pose; returns detections + 17 COCO keypoints + track IDs per frame |
| Person Tracker (fallback) | `app/vision/person_tracker.py` | Dependency-free greedy IOU tracker, used only if a pose backend has no native tracker |
| Keypoint schema | `app/pose/keypoints.py` | `Pose`/`Keypoint` data structures, COCO-17 layout, skeleton edges for drawing |
| Joint angles | `app/pose/angles.py` | Reusable `calculate_angle(a,b,c)` primitive + the 6 required joint/torso angles |
| Normalization | `app/pose/normalization.py` | Hip-centered, torso-scaled keypoints (position/scale invariant) + a separate `estimate_scale()` used for *velocity* signals (see note in section 4) |
| Sequence buffer | `app/pose/sequence.py` | Rolling per-person pose history; motion/velocity/stability helpers |
| Activity detectors | `app/activities/*.py` | One rule module per activity (`standing`, `sitting`, `walking`, `hand_raise`, `bending`, `squatting`, `fall`) |
| State machine | `app/activities/base_activity.py` | Candidate → Confirmed → Active → Ended smoothing, shared by every detector |
| Activity Manager | `app/events/activity_manager.py` | Orchestrates all of the above per tracked person; resolves conflicting activities by priority; drives fall lifecycle, squat counting, unsafe-bending, timeline, DB writes |
| Alert Engine | `app/events/alerts.py` | Cooldown-gated alert firing |
| Evidence | `app/events/evidence.py` | Saves full-frame / crop evidence images |
| Database | `app/database/database.py` | SQLite schema + CRUD + CSV export |
| Dashboard | `app/dashboard/dashboard.py` | Streamlit live analytics, history, filters, CSV export |
| Application entry point | `app/main.py` | Wires everything together for live/recorded video |

## 3. Why YOLO-Pose

- One model performs detection + 17-point COCO pose estimation in a
  single forward pass, collapsing two pipeline stages into one call.
- Ships with ByteTrack/BoT-SORT built in (`model.track(..., persist=True)`),
  satisfying the persistent-ID tracking requirement without a separate
  library.
- The nano checkpoint (`yolov8n-pose.pt`) runs in real time on CPU;
  larger checkpoints (`s`/`m`) trade FPS for accuracy if a GPU is
  available.
- Known limitations: struggles with heavy occlusion, very small/distant
  people, unusual poses (e.g. someone lying flat with the camera
  side-on), and low light. Documented further in `docs/activity_rules.md`.

## 4. A design note worth reading before extending this code

`normalize_pose()` centers every keypoint on the hip midpoint. That is
*exactly right* for shape/posture comparisons (e.g. "is the knee bent
90 degrees?") but it also means the hip's own coordinates are always
`(0, 0)` after normalization — so hip *translation* (the signal walking
and fall-onset detection need) cannot be measured from the normalized
pose. `estimate_scale()` in `normalization.py` and the motion helpers in
`sequence.py` (`hip_motion`, `keypoint_velocity`) deliberately read the
**raw** pose coordinates and divide by a torso-length scale factor
instead, which keeps the measurement resolution/distance-invariant
while preserving translation. This distinction is easy to get wrong —
it originally caused walking and fall sudden-onset detection to always
read zero motion during development, caught by `tests/test_activities.py`
and `tests/test_fall.py`.

## 5. Multi-person handling (Advanced Feature — Section 35)

Each tracked person gets an independent `PersonState` (pose history,
per-activity state-machine runtimes, timeline, squat counter, fall
event, active alerts) keyed by track ID in `ActivityManager.people`.
Every frame, `ActivityManager.process_person()` is called once per
detected person, so people are never cross-contaminated. Stale tracks
(`expire_stale_people`) are dropped after `track_expiry_seconds` of no
detections, closing any open activity/timeline entry first.
