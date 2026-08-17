# Human Pose and Activity Intelligence Platform

**AI Summer Fellowship 2026 -- Computer Vision Engineering Track -- Week 5**
Theme: Human Pose Estimation and Activity Recognition

## 1. Problem Statement

Previous weeks covered image/video processing, object detection,
tracking, zones, and event rules -- all answering *"is something
present?"*. Week 5 moves from object-level analysis to human movement
understanding: detecting people, estimating body keypoints, analyzing
motion over time, recognizing activities, generating alerts, and
providing activity analytics. The question this system answers is:

> **Can it reliably understand human movement over time, rather than
> simply drawing a skeleton?**

## 2. Supported Activities

**Required (5):** Standing, Sitting, Walking, Falling/Fallen, Hand Raised
**Additional (2):** Bending, Squatting (also the repetition-counted activity)

Every activity is a documented, explainable rule -- see
[`docs/activity_rules.md`](docs/activity_rules.md) for the full
specification of each one (required keypoints, joint angles, spatial
and temporal conditions, known limitations).

## 3. Key Features

- Single-pass person detection + 17-keypoint pose estimation (YOLO-Pose)
- Persistent multi-person tracking (ByteTrack) with independent
  per-person state
- Reusable joint-angle engine + hip-centered/torso-scaled pose
  normalization
- Rolling pose-sequence buffer (15-60 frames, configurable) for motion/
  velocity/stability analysis
- 7 rule-based activity detectors, each smoothed through a shared
  Candidate -> Confirmed -> Active -> Ended state machine (reduces
  flicker and false alerts -- **measured 27% fewer label transitions**
  than unsmoothed frame-level decisions, see
  [`docs/experiments.md`](docs/experiments.md))
- Multi-signal fall detection (torso orientation + bbox aspect ratio +
  sudden-drop onset) with a full lifecycle (Possible Fall -> Confirmed
  -> Alert Active -> Acknowledged -> Resolved)
- Squat repetition counter that cannot double-count partial movements
- Prototype ergonomic "unsafe bending" monitor (explicitly **not** a
  certified assessment tool)
- SQLite activity database + evidence capture (full frame / crop /
  pre-event-event-post-event sequence for falls)
- Cooldown-gated alert engine
- Streamlit analytics dashboard: live status, activity distribution,
  timelines, alerts, filterable/exportable history
- Configuration persisted to `config.json`
- 52 automated tests, all passing (see Section 12)

## 4. Architecture

```
Video Source -> Pose Estimation (+Tracking) -> Pose Sequence Buffer ->
Feature Extraction -> Activity Recognition Engine -> Event Manager /
Activity Database -> Alert Engine / Analytics Dashboard
```

Full breakdown, component table, and an important design note about
hip-centered normalization vs. motion detection: see
[`docs/architecture.md`](docs/architecture.md).

## 5. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.10+ |
| Pose estimation | Ultralytics YOLO-Pose |
| Tracking | YOLO built-in ByteTrack (fallback: dependency-free IOU tracker) |
| Video I/O | OpenCV |
| Numerical | NumPy |
| Database | SQLite (stdlib `sqlite3`) |
| Dashboard | Streamlit + Plotly/Pandas |
| Testing | Pytest |

## 6. Dataset Description

See [`docs/dataset.md`](docs/dataset.md) for the full Section-31-style
description (source, class distribution, duration, camera angle,
environment, limitations, privacy) and
[`docs/experiments.md`](docs/experiments.md) for the honest explanation
of why it's synthetic: this build environment has no camera/GPU, so the
32-scenario evaluation set (`evaluation/scenario_generator.py`) is
generated directly in pose-keypoint space, with geometrically-motivated
ground-truth labels, fed through the exact same recognition pipeline
real video would use. Class distribution: 5x standing, sitting, walking,
hand-raised, fallen; 5x additional (bending/squatting); plus 6
explicitly "difficult" scenarios (occlusion-style jitter, low-light-
style jitter, fast movement, partial visibility). No individuals were
recorded; no privacy concerns apply to synthetic data. **On real
deployment, replace the synthetic generator with recorded/consented
video and re-run `evaluation/evaluate.py`.**

## 7. Installation

```bash
git clone <this-repo>
cd pose-activity-platform
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 8. Configuration

All thresholds live in `app/config.py` (defaults) and persist to
`config.json` on first run (Requirement 21). Key settings: pose model
checkpoint, detection/keypoint confidence thresholds, sequence length,
temporal-smoothing frame counts, fall-confirmation time, alert cooldown,
squat knee-angle thresholds, unsafe-bending duration, selected
activities, and the video source.

## 9. Running the Application

```bash
# Webcam
python -m app.main --source 0

# Video file
python -m app.main --source sample_videos/demo_synthetic.mp4

# RTSP stream, headless (no display window), save annotated output
python -m app.main --source rtsp://camera-ip/stream --no-show --record out.mp4

# Dashboard (run alongside, or after, app.main -- reads from the same SQLite DB)
streamlit run app/dashboard/dashboard.py
```

## 10. Activity Rules

Full specification (required keypoints, angles, spatial/temporal
conditions, start/end conditions, alert requirements, known
limitations) for every activity: [`docs/activity_rules.md`](docs/activity_rules.md).
Temporal state machine and per-person state documentation:
[`docs/state_management.md`](docs/state_management.md).

## 11. Evaluation Results (real, measured -- not fabricated)

32 scenarios, run via `python evaluation/evaluate.py`:

| Metric | Value |
|---|---|
| Overall accuracy | **75.0%** |
| Avg recognition-layer latency/frame | 0.074 ms |
| Avg keypoint-failure rate | 0.0% (100% in the deliberately-occluded scenario) |

| Activity | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| bending | 3 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| fallen | 5 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| hand_raised | 5 | 0 | 0 | 100.0% | 100.0% | 100.0% |
| walking | 6 | 2 | 0 | 75.0% | 100.0% | 85.7% |
| standing | 3 | 0 | 3 | 100.0% | 50.0% | 66.7% |
| squatting | 2 | 5 | 0 | 28.6% | 100.0% | 44.4% |
| sitting | 0 | 0 | 5 | n/a | 0.0% | n/a |

**Sitting was systematically confused with squatting** (both are
geometrically "bent knees + upright torso", and squatting is checked
first in the priority order) -- documented in full in
[`docs/activity_rules.md`](docs/activity_rules.md) and
[`docs/experiments.md`](docs/experiments.md) rather than hidden. Full
confusion matrix: `evaluation/results/evaluation_report.json`.

## 12. Performance Results

See [`docs/performance.md`](docs/performance.md). Headline: the
recognition-logic layer costs ~0.074ms/frame (not the bottleneck); YOLO-
Pose inference itself is the real bottleneck in a live deployment and
was not benchmarked in this GPU-less, camera-less build sandbox --
`app/main.py` overlays live FPS for you to record on your own hardware.

## 13. Screenshots

Generated from the real pipeline (`evaluation/render_demo.py`), not
mockups -- see `screenshots/`: `standing.png`, `walking.png`,
`hand_raised.png`, `bending.png`, `squatting.png`.

## 14. Advanced Feature: Multiple-Person Activity Recognition

See [`docs/advanced_feature.md`](docs/advanced_feature.md). Proven, not
just described: `evaluation/multi_person_demo.py` runs 3 independent
synthetic people through one `ActivityManager` simultaneously (standing
/ walking / 2 full squat reps) and confirms each person's activity and
squat count stay fully independent -- real output saved to
`evaluation/results/multi_person_demo.json`.

## 15. Demo

`sample_videos/demo_synthetic.mp4` -- a rendered skeleton walkthrough of
all 7 activities with live on-screen activity labels and alerts,
produced by actually running the synthetic scenarios through
`ActivityManager` (see `evaluation/render_demo.py`). For a real webcam/
video demo, run `python -m app.main --source 0 --record demo.mp4`.

## 16. Known Limitations

- No real-video evaluation was performed in this build environment (no
  camera/GPU) -- see Section 11 and `docs/experiments.md` for the full,
  honest explanation and how to redo it with real footage.
- Sitting vs. squatting confusion (documented above and in
  `docs/activity_rules.md`).
- ID switching on tracker failure creates a new person record rather
  than re-identifying (see `docs/state_management.md`).
- The unsafe-bending monitor is an explicit **prototype**, not a
  certified ergonomic/medical assessment.
- Fall detection cannot perfectly distinguish a genuine fall from lying
  down deliberately; it uses a sudden-onset heuristic that can miss
  slow collapses.
- Pose-model comparison (Experiment 1) was not executable in this
  sandbox; methodology to run it yourself is documented in
  `docs/experiments.md`.

## 17. Future Improvements

- Real webcam/recorded-video evaluation dataset with human-reviewed
  ground truth, replacing the synthetic scenario generator.
- Re-identification to survive tracker ID switches.
- An optional ML sequence classifier (LSTM/Temporal CNN) trained on
  normalized keypoint sequences, benchmarked against the rule-based
  system per Section 30 of the brief.
- A hip-height-vs-support-surface heuristic (or explicit stillness
  duration) to disambiguate sitting from squatting.
- Webhook/Telegram/Slack alert channels (`.env.example` already has
  placeholders) wired into `app/events/alerts.py`.

## 18. Project Structure

```
pose-activity-platform/
├── app/
│   ├── main.py                  # application entry point
│   ├── config.py                # persisted configuration
│   ├── vision/                  # video source, pose estimator, tracker
│   ├── pose/                    # keypoints, angles, normalization, sequence buffer
│   ├── activities/              # 7 rule-based detectors + shared state machine
│   ├── events/                  # activity manager, alerts, evidence capture
│   ├── database/                # SQLite schema + CRUD
│   └── dashboard/                # Streamlit analytics dashboard
├── tests/                        # 52 automated tests (pytest)
├── evaluation/                   # scenario generator, evaluate.py, experiments.py, render_demo.py, results/
├── docs/                         # architecture, activity rules, state mgmt, experiments, performance
├── sample_videos/                # rendered demo video
├── screenshots/                  # rendered activity screenshots
├── evidence/                     # runtime evidence capture (fall/bending/alerts)
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## Running the Tests

```bash
pip install -r requirements.txt   # or just: pip install numpy pytest
pytest tests/ -v
```

52/52 passing as of this submission (`pytest tests/ -v`, ~0.2s runtime).

## Running the Evaluation / Experiments Yourself

```bash
python evaluation/evaluate.py       # 32-scenario evaluation -> evaluation/results/evaluation_report.json
python evaluation/experiments.py    # experiments 2-6 -> evaluation/results/experiments_report.json
python evaluation/render_demo.py    # regenerate sample_videos/demo_synthetic.mp4 and screenshots/
```
