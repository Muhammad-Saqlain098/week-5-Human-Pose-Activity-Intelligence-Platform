"""
Evaluation harness (Requirement 33).

Runs every scenario from scenario_generator.py through the REAL
ActivityManager / activity-detector pipeline (the same code path used by
app/main.py on live video), then computes:
    - per-class true/false positives/negatives, precision, recall, F1
    - a confusion matrix
    - keypoint-failure rate (frames with < 50% keypoints visible)
    - average per-scenario processing latency

No numbers here are invented -- everything is measured by actually
running the code. See scenario_generator.py for the documented
limitation this evaluation set is synthetic pose data, not real video.
"""
from __future__ import annotations
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.events.activity_manager import ActivityManager
from evaluation.scenario_generator import build_evaluation_set, Scenario

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
LABELS = ["standing", "sitting", "walking", "hand_raised", "fallen", "bending", "squatting", "none"]


def run_scenario(scenario: Scenario, config: Config) -> dict:
    manager = ActivityManager(config, db=None, source_id="eval")
    activity_hit_counts = defaultdict(int)
    keypoint_failures = 0
    total_frames = len(scenario.frames)

    t0 = time.perf_counter()
    for pose, ts in scenario.frames:
        if pose.visible_ratio(config.keypoint_conf_threshold) < 0.5:
            keypoint_failures += 1
        state = manager.process_person(pose, frame=None, timestamp=ts)
        if state.current_activity:
            activity_hit_counts[state.current_activity] += 1
    elapsed = time.perf_counter() - t0

    predicted = max(activity_hit_counts, key=activity_hit_counts.get) if activity_hit_counts else "none"

    return {
        "scenario": scenario.name,
        "difficulty": scenario.difficulty,
        "ground_truth": scenario.label,
        "predicted": predicted,
        "activity_hit_counts": dict(activity_hit_counts),
        "keypoint_failure_rate": keypoint_failures / total_frames if total_frames else 0.0,
        "total_frames": total_frames,
        "latency_per_frame_ms": (elapsed / total_frames * 1000) if total_frames else 0.0,
    }


def compute_metrics(results: list) -> dict:
    labels = sorted(set([r["ground_truth"] for r in results] + [r["predicted"] for r in results]))
    confusion = {gt: {pred: 0 for pred in labels} for gt in labels}
    for r in results:
        confusion[r["ground_truth"]][r["predicted"]] += 1

    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fn = sum(confusion[label][p] for p in labels if p != label)
        fp = sum(confusion[gt][label] for gt in labels if gt != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)) if precision and recall and (precision + recall) > 0 else None
        per_class[label] = {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}

    total = len(results)
    correct = sum(1 for r in results if r["ground_truth"] == r["predicted"])
    overall_accuracy = correct / total if total else 0.0

    return {"labels": labels, "confusion_matrix": confusion, "per_class": per_class,
            "overall_accuracy": overall_accuracy, "total_scenarios": total}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    config = Config.load()
    scenarios = build_evaluation_set()

    print(f"Running {len(scenarios)} evaluation scenarios...")
    results = [run_scenario(s, config) for s in scenarios]
    metrics = compute_metrics(results)

    avg_latency = sum(r["latency_per_frame_ms"] for r in results) / len(results)
    avg_kp_failure = sum(r["keypoint_failure_rate"] for r in results) / len(results)

    report = {
        "num_scenarios": len(scenarios),
        "overall_accuracy": metrics["overall_accuracy"],
        "per_class": metrics["per_class"],
        "confusion_matrix": metrics["confusion_matrix"],
        "avg_latency_per_frame_ms": avg_latency,
        "avg_keypoint_failure_rate": avg_kp_failure,
        "raw_results": results,
    }

    out_path = os.path.join(RESULTS_DIR, "evaluation_report.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nOverall accuracy: {metrics['overall_accuracy']*100:.1f}%")
    print(f"Avg latency/frame: {avg_latency:.4f} ms | Avg keypoint failure rate: {avg_kp_failure*100:.1f}%")
    print("\nPer-class results:")
    for label, m in metrics["per_class"].items():
        p = f"{m['precision']*100:.1f}%" if m["precision"] is not None else "n/a"
        r = f"{m['recall']*100:.1f}%" if m["recall"] is not None else "n/a"
        f1 = f"{m['f1']*100:.1f}%" if m["f1"] is not None else "n/a"
        print(f"  {label:12s} TP={m['tp']:2d} FP={m['fp']:2d} FN={m['fn']:2d}  P={p:>6s} R={r:>6s} F1={f1:>6s}")

    print(f"\nFull report saved to {out_path}")
    return report


if __name__ == "__main__":
    main()
