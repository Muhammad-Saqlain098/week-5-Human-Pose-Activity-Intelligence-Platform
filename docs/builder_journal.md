# Builder Journal (Week 5)

## What I built

A complete Human Pose and Activity Intelligence Platform: YOLO-Pose
based person detection + keypoint estimation, persistent multi-person
tracking, a reusable joint-angle/normalization/sequence-buffer feature
layer, 7 rule-based activity detectors (standing, sitting, walking,
hand-raised, fallen, bending, squatting) smoothed by a shared temporal
state machine, a multi-signal fall-detection lifecycle, a squat
repetition counter, a prototype unsafe-bending ergonomic monitor,
SQLite persistence with evidence capture, a cooldown-gated alert engine,
a Streamlit analytics dashboard, 52 passing automated tests, and a real
(if synthetic) 32-scenario evaluation with a genuine confusion matrix
and six documented experiments.

## Most difficult pose-estimation problem

Not the estimation itself (YOLO-Pose is well-documented) but a subtle
bug in how I combined normalization with motion detection: I originally
computed hip-motion and fall sudden-onset from the *hip-centered
normalized* keypoints, which by construction always place the hip at
`(0,0)` -- so walking and fall-onset detection silently always measured
zero motion. Two of my own tests caught this
(`test_walking_true_with_hip_and_ankle_motion`,
`test_sudden_drop_detection`), which forced a redesign: motion/velocity
now reads raw pose coordinates and divides by a separately-computed
torso-length scale factor, keeping resolution-invariance without
discarding translation. This is documented explicitly in
`docs/architecture.md` so I (or anyone else) doesn't reintroduce it.

## Activities that were easiest to detect

Hand Raised (a single clear geometric signal, wrist above shoulder) and
Fallen (a strong, hard-to-confuse-with-anything-else horizontal/wide
bounding-box signal) both hit 100% precision/recall on the evaluation
set.

## Activities that were hardest to detect

Sitting -- it shares almost the exact geometric signature as Squatting
(bent knees, upright torso), and because Squatting is checked earlier in
the priority order, every sitting scenario in evaluation was resolved
as "squatting" (0% sitting recall, 28.6% squatting precision). This is
a genuine, measured limitation, not something I'm inferring -- see
`docs/experiments.md`.

## False alerts encountered

None from a real camera (not available in this environment), but the
experiments harness surfaced the *cause* of likely false alerts: with
temporal smoothing disabled (frame-level decisions), the average number
of activity-label transitions per scenario rose ~27% (1.16 -> 1.47),
which is exactly the kind of flicker that would translate into
duplicate/false alerts on noisy real video. This measured result is why
the assignment insists on the Candidate/Confirmed/Active/Ended state
machine rather than reporting raw per-frame rule output.

## How camera angle affected results

I couldn't render true multi-angle 3D footage in this sandbox, so I used
scenario "difficulty" (increasing synthetic keypoint jitter, and a
deliberately-occluded lower body) as a documented proxy. Accuracy
degraded smoothly with jitter (84.2% normal -> 66.7% occlusion-level ->
60.0% low-light-level) and collapsed to 0% when leg keypoints were
zeroed out, since standing/sitting/squatting/walking all depend on
knee/hip/ankle visibility. In a real deployment, a poor camera angle
(e.g. steep overhead, or a person mostly out of frame) would produce
exactly this kind of partial-keypoint failure.

## How temporal logic improved performance

Directly measured in Experiment 4: smoothing didn't change final
accuracy on our (clean) synthetic set, but cut label instability by
~27%, which is the metric that actually matters for avoiding duplicate
alerts and flickering dashboard labels on real, noisy video.

## Main system bottleneck

The recognition-logic layer measured at ~0.074ms/frame -- effectively
free. The real bottleneck in any live deployment is the YOLO-Pose
forward pass itself, which I could not benchmark here (no GPU, no
camera). `app/main.py` already overlays live FPS so this is trivial to
measure on real hardware.

## Most useful experiment

Experiment 5 (fall confirmation time): raising the confirmation window
from 0.6s to 1.2s dropped fall recall to 0% because our scenarios simply
aren't long enough to confirm at that threshold -- a concrete, numeric
illustration of the detection-delay vs. false-alarm trade-off the
brief asks fellows to characterize, rather than a hand-wavy description
of it.

## What I would redesign

The activity priority-resolution order (`fallen > hand_raised >
squatting > bending > sitting > walking > standing`) is a single global
list; sitting vs. squatting disambiguation would benefit from either a
dedicated "was this knee-bend accompanied by hip-height stability over
2+ seconds" signal (squats are inherently transient; sitting is
sustained) rather than relying purely on priority order.

## Goals for Week 6

- Replace the synthetic evaluation set with real recorded/consented
  video and re-run the full evaluation + experiment suite.
- Add the sitting/squatting disambiguation heuristic described above.
- Wire at least one real alert channel (webhook or Telegram) using the
  placeholders already in `.env.example`.
- Explore the optional ML sequence classifier (Section 30) and compare
  it against the rule-based system on the same evaluation set.
