"""
Integration tests for the ActivityManager: full per-frame pipeline,
alert cooldown, tracking expiry, and invalid/missing-keypoint handling.
"""
import time
import os
import tempfile
from app.config import Config
from app.database.database import ActivityDatabase
from app.events.activity_manager import ActivityManager
from app.pose.keypoints import Pose, Keypoint
from tests.conftest import make_pose


def _tmp_db_path():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


def test_activity_manager_confirms_standing_over_frames():
    config = Config()
    config.candidate_to_confirmed_frames = 3
    db_path = _tmp_db_path()
    db = ActivityDatabase(db_path)
    manager = ActivityManager(config, db=db)

    pose = make_pose(person_id=1)
    t = 0.0
    for _ in range(6):
        manager.process_person(pose, frame=None, timestamp=t)
        t += 0.1

    state = manager.people[1]
    assert state.current_activity == "standing"
    db.close()
    os.remove(db_path)


def test_activity_manager_handles_missing_keypoints_gracefully():
    config = Config()
    db = ActivityDatabase(_tmp_db_path())
    manager = ActivityManager(config, db=db)

    # a pose with almost no visible keypoints must not crash the pipeline
    sparse_pose = Pose(person_id=9, keypoints={"nose": Keypoint(10, 10, 0.9)})
    state = manager.process_person(sparse_pose, frame=None, timestamp=0.0)
    assert state.current_activity is None  # no rule can confirm with this little data
    db.close()


def test_tracking_expiry_removes_stale_person():
    config = Config()
    config.track_expiry_seconds = 1.0
    db = ActivityDatabase(_tmp_db_path())
    manager = ActivityManager(config, db=db)

    pose = make_pose(person_id=1)
    manager.process_person(pose, frame=None, timestamp=0.0)
    assert 1 in manager.people
    expired = manager.expire_stale_people(current_time=5.0)
    assert 1 in expired
    assert 1 not in manager.people
    db.close()


def test_alert_cooldown_suppresses_repeated_alerts():
    from app.events.alerts import AlertEngine
    engine = AlertEngine(cooldown_seconds=10.0)
    a1 = engine.fire(person_id=1, alert_type="fall_detected", timestamp=0.0, message="fall!")
    a2 = engine.fire(person_id=1, alert_type="fall_detected", timestamp=2.0, message="fall again!")
    a3 = engine.fire(person_id=1, alert_type="fall_detected", timestamp=11.0, message="fall once more")
    assert a1 is not None
    assert a2 is None  # suppressed: within cooldown window
    assert a3 is not None  # cooldown expired


def test_activity_timeline_records_duration():
    config = Config()
    config.candidate_to_confirmed_frames = 2
    config.end_grace_frames = 1
    db = ActivityDatabase(_tmp_db_path())
    manager = ActivityManager(config, db=db)

    pose = make_pose(person_id=3)
    for i in range(4):
        manager.process_person(pose, frame=None, timestamp=float(i))

    state = manager.people[3]
    assert len(state.timeline) >= 1
    assert state.timeline[0].activity == "standing"
    db.close()
