# Performance Report

## 1. What was actually measured

The activity-recognition pipeline (feature extraction -> rule
evaluation -> temporal smoothing -> DB write) was profiled directly by
`evaluation/evaluate.py` while running all 32 evaluation scenarios
(~700 total frames):

| Metric | Measured value |
|---|---|
| Average processing latency per frame (recognition logic only, no pose estimation) | **0.074 ms** |
| Implied max throughput of the recognition layer alone | **> 13,000 FPS** (i.e. not the bottleneck) |
| Average keypoint-failure rate (frames with < 50% visible keypoints) across the full evaluation set | 0.0% (rises to 100% in the deliberately-occluded `partial_visibility` scenario -- see `docs/experiments.md`) |

These numbers were captured on the CPU-only sandbox used to build this
project (no GPU); see `evaluation/results/evaluation_report.json` ->
`raw_results[].latency_per_frame_ms` for the per-scenario breakdown.

## 2. Main bottleneck

**The pose estimation model (YOLO-Pose), not the activity-recognition
logic, is the bottleneck in a real deployment.** The measurements above
show the rule engine, state machine, and database writes cost a
fraction of a millisecond per frame -- effectively free compared to a
neural network forward pass. In a live deployment the frame budget is
dominated by:

1. Video decode (cv2.VideoCapture read)
2. YOLO-Pose forward pass (detection + keypoints in one call)
3. Tracker association (ByteTrack, comparatively cheap)
4. **Everything measured above** (comparatively negligible)
5. Frame draw/encode (if `--record` or `--show` is used)

On CPU, `yolov8n-pose.pt` typically runs in the 15-30+ FPS range at
640px input on modern hardware (not independently benchmarked in this
sandbox, which has no compatible GPU/webcam); `yolov8s`/`yolov8m` trade
FPS for accuracy. A GPU is recommended for multi-camera or higher-
resolution deployments.

## 3. What was NOT measured here (and why)

- **End-to-end FPS with real YOLO-Pose inference**: requires a GPU/CPU
  capable of running Ultralytics + a real or webcam video source, which
  this build sandbox does not have (no camera, network egress limited to
  package registries). `app/main.py` already overlays live FPS on
  screen -- this is the number to record on your own machine.
- **Memory usage under load**: not profiled with `psutil` in this
  environment; the pose-sequence buffer's memory footprint is small and
  bounded (`sequence_length` x keypoints x floats per tracked person),
  so it scales linearly and predictably with the number of concurrently
  tracked people.
- **GPU utilization**: optional per the brief; not applicable without a
  GPU in this environment.

## 4. Practical guidance for the demo / viva

When running `python -m app.main --source 0`, watch the on-screen FPS
counter and record:
- Person-detection + pose-estimation time (dominates total frame time)
- Total frame-processing time (detection + tracking + recognition + draw)
- CPU usage (Task Manager / `top`) during a multi-person scene

These are the four numbers Requirement 40 asks for; the recognition-
layer numbers above (already measured) satisfy the "activity-analysis
time" portion of that requirement independently of what hardware you
demo on.
