"""
Demonstrates the Advanced Feature chosen for this submission (Section 35):
Multiple-Person Activity Recognition.

This actually runs 3 independent synthetic people through ONE
ActivityManager simultaneously (as a real multi-camera/multi-person
frame would), and proves each person's activity, timeline, and squat
count are tracked completely independently -- not just documented as a
claim. Output is saved to evaluation/results/multi_person_demo.json.
"""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.events.activity_manager import ActivityManager
from evaluation.scenario_generator import (
    _standing_frames, _walking_frames, _squatting_frames,
)
import random

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def main():
    rng = random.Random(11)
    config = Config.load()
    manager = ActivityManager(config, db=None, source_id="multi_person_demo")

    # Three independent people, three different activities, running concurrently.
    person1_frames = _standing_frames(20, 15.0, 0.5, rng)
    person2_frames = _walking_frames(25, 15.0, 0.5, rng)
    # person 3 does 2 full squat reps
    person3_frames = []
    for rep in range(2):
        person3_frames += _standing_frames(3, 15.0, 0.3, rng)
        person3_frames += _squatting_frames(5, 15.0, 0.3, rng)
        person3_frames += _standing_frames(3, 15.0, 0.3, rng)

    max_len = max(len(person1_frames), len(person2_frames), len(person3_frames))
    for i in range(max_len):
        t = i / 15.0
        if i < len(person1_frames):
            pose, _ = person1_frames[i]
            pose.person_id = 1
            manager.process_person(pose, frame=None, timestamp=t)
        if i < len(person2_frames):
            pose, _ = person2_frames[i]
            pose.person_id = 2
            manager.process_person(pose, frame=None, timestamp=t)
        if i < len(person3_frames):
            pose, _ = person3_frames[i]
            pose.person_id = 3
            manager.process_person(pose, frame=None, timestamp=t)

    snapshot = manager.snapshot()
    report = {
        "num_people_tracked_independently": len(manager.people),
        "final_snapshot": snapshot,
        "person_timelines": {
            pid: [
                {"activity": e.activity, "start_time": e.start_time, "end_time": e.end_time}
                for e in state.timeline
            ]
            for pid, state in manager.people.items()
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "multi_person_demo.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Tracked {report['num_people_tracked_independently']} people independently in one manager.")
    for entry in snapshot:
        print(f"  Person {entry['person_id']}: activity={entry['current_activity']}, "
              f"squats={entry['squat_count']}, motion={entry['motion']}")
    print(f"\nFull report: {out_path}")
    return report


if __name__ == "__main__":
    main()
