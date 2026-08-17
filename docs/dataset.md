# Dataset Description (Section 31)

## Approach taken, and why

This submission uses **rule-based** activity recognition (as explicitly
permitted by the brief), so no training dataset is strictly required --
the activity rules in `docs/activity_rules.md` are hand-specified from
joint-angle geometry, not learned. The "dataset" that matters for this
system is therefore the **evaluation set**, used to measure the rules'
accuracy (Section 32), not a training set.

Because this build environment has no camera and could not record real
video (see the honest limitation notes in `docs/experiments.md`), the
evaluation set is generated synthetically and directly in pose-keypoint
space by `evaluation/scenario_generator.py`, with every scenario's
ground-truth label determined by the actual geometry used to construct
it (e.g. a "standing" scenario really does have straight knees and an
upright torso in its coordinates -- the label isn't just an assumption).

## Dataset source

`evaluation/scenario_generator.py` -- procedurally generated pose
sequences, not sourced from any external dataset, public clip, or
recording of a real person. No privacy or consent concerns apply, since
no real individual's image or video is used anywhere in this dataset.

## Class distribution

| Class | # scenarios | Frames per scenario | Notes |
|---|---|---|---|
| standing | 5 (+1 partial-visibility variant) | 25 | 1 extra "difficult" variant with legs occluded |
| sitting | 5 | 25 | |
| walking | 5 (+1 fast-movement variant) | 30 (15 for the fast variant, at 2x fps) | |
| hand_raised | 5 | 15 | |
| fallen | 5 | 20 | includes a 4-frame "standing" lead-in to test sudden-onset detection |
| bending | 3 | 25 | 1 occlusion-jitter variant |
| squatting | 2 | 25 | |
| **Total** | **32 scenarios**, ~700 frames | | exceeds the required minimum of 30 |

Each of the five "required" activity groups includes a mix of
`difficulty` tags (`normal`, `occlusion`, `low_light`) via a jitter
parameter added to every keypoint coordinate, simulating unstable/noisy
keypoint estimates. Two additional explicit difficulty scenarios are
included beyond the base 30: `walking_fast` (time-compressed motion) and
`standing_partial_visibility` (lower-body keypoints deliberately zeroed
out), directly covering the "difficult examples" the brief requires
(occlusion, poor lighting proxy, partial body visibility, fast
movement). Camera-angle variation and true multi-person frames are not
covered by this synthetic set (documented as a limitation in
`docs/experiments.md`, Experiment 6) but multi-person handling itself is
proven separately in `docs/advanced_feature.md`.

## Video duration

Not applicable in frame-count terms the way a real video file would be
-- each scenario is 15-30 synthetic frames at an assumed 15fps (~1-2
seconds), matching the short duration a real "hand raised" or "fall"
event would actually take.

## Camera angle / Environment

Not applicable -- scenarios are generated directly in 2D keypoint
coordinate space rather than rendered from a 3D scene, so there is no
camera position or background environment to vary. This is the
project's most significant, explicitly-documented limitation (see
`docs/experiments.md`, Experiment 1 and Experiment 6).

## Limitations

- Entirely synthetic; does not capture real YOLO-Pose keypoint noise,
  motion blur, real occlusion patterns, or genuine human movement
  variability.
- No true camera-angle or multi-person-in-frame variation.
- Ground-truth labels are geometrically self-consistent by construction
  (since the scenario generator and the activity rules both reason about
  the same joint-angle geometry), which likely **overstates** real-world
  accuracy for activities whose rule design happens to match the
  generator's assumptions closely (e.g. hand_raised, fallen) and may
  **understate** it for classes with real-world variability the
  generator doesn't model.

## Privacy considerations

None apply -- no real people, images, or video were recorded or used.
If this system is deployed with real cameras, **do not record
individuals without consent**, per the brief's explicit instruction, and
follow local regulations regarding video surveillance and biometric
data (pose/gait can be considered biometric data in some
jurisdictions).

## Recommended next step (Week 6)

Replace this synthetic set with a real, consented recording covering the
required 30+ scenarios across genuine camera angles, lighting, and
multiple people, then re-run `evaluation/evaluate.py` and
`evaluation/experiments.py` unchanged -- the harness itself does not
need to change, only the source of `Pose` objects.
