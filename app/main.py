"""
Application entry point.

Wires together the full pipeline required by the assignment:

  Video Source -> Pose Estimation (+Tracking) -> Pose Sequence Buffer ->
  Feature Extraction -> Activity Recognition Engine -> Event Manager /
  Activity Database -> Alert Engine / Analytics Dashboard

Usage:
    python -m app.main --source 0                  # webcam
    python -m app.main --source sample_videos/demo.mp4
    python -m app.main --source rtsp://... --no-show
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import Config
from app.database.database import ActivityDatabase
from app.events.activity_manager import ActivityManager
from app.pose.keypoints import SKELETON_EDGES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("main")


def draw_overlay(frame, poses, manager, show_skeleton: bool):
    import cv2
    for pose in poses:
        x1, y1, x2, y2 = [int(v) for v in pose.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)

        if show_skeleton:
            for a, b in SKELETON_EDGES:
                ka, kb = pose.keypoints.get(a), pose.keypoints.get(b)
                if ka and kb and ka.confidence > 0.3 and kb.confidence > 0.3:
                    cv2.line(frame, (int(ka.x), int(ka.y)), (int(kb.x), int(kb.y)), (255, 180, 0), 2)
            for kp in pose.keypoints.values():
                if kp.confidence > 0.3:
                    cv2.circle(frame, (int(kp.x), int(kp.y)), 3, (0, 0, 255), -1)

        state = manager.people.get(pose.person_id)
        label = f"ID {pose.person_id}: {state.current_activity or '...'}" if state else f"ID {pose.person_id}"
        cv2.putText(frame, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2)
        if state and state.active_alerts:
            cv2.putText(frame, f"ALERT: {','.join(state.active_alerts)}", (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return frame


def run(args):
    config = Config.load()
    if args.source:
        config.default_source = args.source
    if args.no_skeleton:
        config.show_skeleton = False

    from app.vision.video_source import VideoSource
    from app.vision.pose_estimator import YoloPoseEstimator

    db = ActivityDatabase(config.db_path)
    manager = ActivityManager(config, db=db, source_id=str(config.default_source))
    estimator = YoloPoseEstimator(config.pose_model, config.detection_conf_threshold, config.tracker)

    writer = None
    try:
        import cv2
        with VideoSource(config.default_source, config.frame_width, config.frame_height) as vs:
            start_time = time.time()
            frame_count = 0
            for idx, frame in vs.frames():
                ts = time.time()
                poses = estimator.infer(frame, frame_index=idx, timestamp=ts)
                for pose in poses:
                    manager.process_person(pose, frame=frame, timestamp=ts)
                manager.expire_stale_people(ts)

                frame = draw_overlay(frame, poses, manager, config.show_skeleton)
                frame_count += 1
                fps = frame_count / max(1e-6, time.time() - start_time)
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

                if args.record:
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), vs.fps(), (w, h))
                    writer.write(frame)

                if not args.no_show:
                    cv2.namedWindow("Pose & Activity Intelligence Platform", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Pose & Activity Intelligence Platform", args.window_width, args.window_height)
                    cv2.imshow("Pose & Activity Intelligence Platform", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
    finally:
        if writer is not None:
            writer.release()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Human Pose and Activity Intelligence Platform")
    parser.add_argument("--source", type=str, default=None, help="0 for webcam, or a video file path / RTSP URL")
    parser.add_argument("--no-show", action="store_true", help="Run headless (no display window)")
    parser.add_argument("--no-skeleton", action="store_true", help="Disable skeleton drawing overlay")
    parser.add_argument("--record", type=str, default=None, help="Path to save an annotated output video")
    parser.add_argument("--window-width", type=int, default=960, help="Display window width in pixels")
    parser.add_argument("--window-height", type=int, default=540, help="Display window height in pixels")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
