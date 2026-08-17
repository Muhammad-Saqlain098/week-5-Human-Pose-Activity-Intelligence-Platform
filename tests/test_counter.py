from app.activities.squatting import SquatCounter
from app.pose.angles import JointAngles


class DummyConfig:
    squat_down_knee_angle = 110.0
    squat_up_knee_angle = 160.0


def _angles(knee):
    return JointAngles(left_knee=knee, right_knee=knee)


def test_squat_counter_counts_one_full_rep():
    counter = SquatCounter(DummyConfig())
    counter.update(_angles(170))   # standing
    counter.update(_angles(90))    # down
    counter.update(_angles(170))   # back up -> count
    assert counter.count == 1


def test_squat_counter_does_not_double_count_partial_movement():
    counter = SquatCounter(DummyConfig())
    counter.update(_angles(170))   # standing
    counter.update(_angles(90))    # down
    counter.update(_angles(130))   # partial rise, NOT past squat_up_knee_angle
    counter.update(_angles(95))    # back down without ever fully standing
    counter.update(_angles(170))   # now fully stands -> exactly one rep counted
    assert counter.count == 1


def test_squat_counter_multiple_reps():
    counter = SquatCounter(DummyConfig())
    for _ in range(3):
        counter.update(_angles(170))
        counter.update(_angles(90))
        counter.update(_angles(170))
    assert counter.count == 3


def test_squat_counter_reset():
    counter = SquatCounter(DummyConfig())
    counter.update(_angles(170)); counter.update(_angles(90)); counter.update(_angles(170))
    assert counter.count == 1
    counter.reset()
    assert counter.count == 0
    assert counter.state == "standing"
