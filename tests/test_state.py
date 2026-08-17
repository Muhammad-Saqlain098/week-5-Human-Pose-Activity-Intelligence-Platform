from app.activities.base_activity import ActivityDetector, ActivityRuntime, ActivityState


def always_true(*args, **kwargs):
    return True


def always_false(*args, **kwargs):
    return False


def test_state_progresses_from_idle_to_active():
    detector = ActivityDetector("dummy", lambda *a: True, confirm_frames=3, end_grace_frames=2)
    runtime = ActivityRuntime()
    t = 0.0
    for _ in range(3):
        runtime = detector.update(runtime, True, t)
        t += 1
    assert runtime.state in (ActivityState.CONFIRMED, ActivityState.ACTIVE)


def test_single_frame_flicker_does_not_confirm():
    detector = ActivityDetector("dummy", lambda *a: True, confirm_frames=5, end_grace_frames=2)
    runtime = ActivityRuntime()
    runtime = detector.update(runtime, True, 0.0)   # candidate
    runtime = detector.update(runtime, False, 1.0)  # drops back to idle before confirming
    assert runtime.state == ActivityState.IDLE


def test_active_state_survives_brief_grace_flicker():
    detector = ActivityDetector("dummy", lambda *a: True, confirm_frames=2, end_grace_frames=3)
    runtime = ActivityRuntime()
    t = 0.0
    for _ in range(3):  # confirm -> active
        runtime = detector.update(runtime, True, t); t += 1
    assert runtime.state == ActivityState.ACTIVE
    runtime = detector.update(runtime, False, t); t += 1  # single false frame: within grace
    assert runtime.state == ActivityState.ACTIVE


def test_active_state_ends_after_grace_exceeded():
    detector = ActivityDetector("dummy", lambda *a: True, confirm_frames=2, end_grace_frames=2)
    runtime = ActivityRuntime()
    t = 0.0
    for _ in range(3):
        runtime = detector.update(runtime, True, t); t += 1
    assert runtime.state == ActivityState.ACTIVE
    for _ in range(2):
        runtime = detector.update(runtime, False, t); t += 1
    assert runtime.state == ActivityState.ENDED
