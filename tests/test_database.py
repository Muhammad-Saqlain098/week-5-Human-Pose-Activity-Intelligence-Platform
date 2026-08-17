import os
import tempfile
from app.database.database import ActivityDatabase


def _tmp_db():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    os.remove(path)
    return path


def test_database_creates_schema():
    path = _tmp_db()
    db = ActivityDatabase(path)
    cur = db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row["name"] for row in cur.fetchall()}
    assert {"activity_events", "alerts", "persons"}.issubset(tables)
    db.close()
    os.remove(path)


def test_insert_and_query_activity_event():
    path = _tmp_db()
    db = ActivityDatabase(path)
    event_id = db.insert_activity_event(person_id=1, activity="standing", start_time=0.0, confidence=0.9)
    rows = db.query_events(person_id=1)
    assert len(rows) == 1
    assert rows[0]["activity"] == "standing"
    assert rows[0]["id"] == event_id
    db.close()
    os.remove(path)


def test_close_activity_event_sets_duration():
    path = _tmp_db()
    db = ActivityDatabase(path)
    event_id = db.insert_activity_event(person_id=1, activity="walking", start_time=10.0)
    db.close_activity_event(event_id, 15.0)
    rows = db.query_events(person_id=1)
    assert rows[0]["duration"] == 5.0
    db.close()
    os.remove(path)


def test_query_events_filters_by_activity():
    path = _tmp_db()
    db = ActivityDatabase(path)
    db.insert_activity_event(person_id=1, activity="standing", start_time=0.0)
    db.insert_activity_event(person_id=1, activity="sitting", start_time=1.0)
    rows = db.query_events(activity="sitting")
    assert len(rows) == 1
    assert rows[0]["activity"] == "sitting"
    db.close()
    os.remove(path)


def test_alert_insert_and_query():
    path = _tmp_db()
    db = ActivityDatabase(path)
    db.insert_alert(person_id=2, alert_type="fall_detected", timestamp=1.0, message="test")
    alerts = db.query_alerts(status="active")
    assert len(alerts) == 1
    assert alerts[0]["alert_type"] == "fall_detected"
    db.close()
    os.remove(path)


def test_export_csv_creates_file():
    path = _tmp_db()
    db = ActivityDatabase(path)
    db.insert_activity_event(person_id=1, activity="standing", start_time=0.0)
    csv_path = path + ".csv"
    db.export_events_csv(csv_path)
    assert os.path.exists(csv_path)
    with open(csv_path) as f:
        content = f.read()
    assert "standing" in content
    db.close()
    os.remove(path)
    os.remove(csv_path)
