"""
Streamlit analytics dashboard (Requirement 19, 20).

Run with:
    streamlit run app/dashboard/dashboard.py

Reads directly from the SQLite activity database, so it can run
alongside (or independently of) the live capture process in app/main.py.
"""
from __future__ import annotations
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import streamlit as st

from app.config import Config
from app.database.database import ActivityDatabase

st.set_page_config(page_title="Pose & Activity Intelligence", layout="wide")

config = Config.load()
db_path = config.db_path
if not os.path.isabs(db_path):
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), db_path)

st.title("Human Pose & Activity Intelligence Platform")
st.caption("Live analytics dashboard -- reads from the activity events database.")

if not os.path.exists(db_path):
    st.warning(f"No database found yet at `{db_path}`. Run the application (`python -m app.main`) first, "
               f"or `python evaluation/evaluate.py` to populate sample data.")
    st.stop()

db = ActivityDatabase(db_path)

# ---------------- Current status ----------------
st.header("Current Status")
events = db.query_events()
df = pd.DataFrame([dict(r) for r in events])

col1, col2, col3, col4 = st.columns(4)
active_people = df["person_id"].nunique() if not df.empty else 0
active_alerts = db.query_alerts(status="active")
col1.metric("Tracked People (all time)", active_people)
col2.metric("Total Activities Logged", len(df))
col3.metric("Active/Unacknowledged Alerts", len(active_alerts))
col4.metric("Falls Detected", int((df["activity"] == "fallen").sum()) if not df.empty else 0)

# ---------------- Activity metrics ----------------
st.header("Activity Metrics")
if not df.empty:
    counts = db.count_events_by_activity()
    counts_df = pd.DataFrame(list(counts.items()), columns=["activity", "count"]).sort_values("count", ascending=False)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Activity Distribution")
        st.bar_chart(counts_df.set_index("activity"))
    with c2:
        st.subheader("Average Duration by Activity (s)")
        avg_dur = df.dropna(subset=["duration"]).groupby("activity")["duration"].mean().reset_index()
        st.bar_chart(avg_dur.set_index("activity"))
else:
    st.info("No activity events recorded yet.")

# ---------------- Alerts ----------------
st.header("Alerts")
alerts = db.query_alerts()
if alerts:
    alerts_df = pd.DataFrame([dict(a) for a in alerts])
    st.dataframe(alerts_df, use_container_width=True)
else:
    st.info("No alerts recorded yet.")

# ---------------- Activity History / filters ----------------
st.header("Activity History")
if not df.empty:
    colf1, colf2, colf3 = st.columns(3)
    person_filter = colf1.selectbox("Person ID", ["All"] + sorted(df["person_id"].unique().tolist()))
    activity_filter = colf2.selectbox("Activity", ["All"] + sorted(df["activity"].unique().tolist()))
    date_filter = colf3.text_input("Filter start_time >= (unix timestamp, optional)")

    filtered = df.copy()
    if person_filter != "All":
        filtered = filtered[filtered["person_id"] == person_filter]
    if activity_filter != "All":
        filtered = filtered[filtered["activity"] == activity_filter]
    if date_filter.strip():
        try:
            filtered = filtered[filtered["start_time"] >= float(date_filter)]
        except ValueError:
            st.error("start_time filter must be a numeric unix timestamp")

    st.dataframe(filtered, use_container_width=True)

    csv_path = os.path.join(os.path.dirname(db_path), "activity_history_export.csv")
    if st.button("Export filtered records to CSV"):
        filtered.to_csv(csv_path, index=False)
        st.success(f"Exported to {csv_path}")
        with open(csv_path, "rb") as f:
            st.download_button("Download CSV", f, file_name="activity_history_export.csv")
else:
    st.info("No history to display yet.")

st.caption(f"Data source: {db_path} | Refreshed at {time.strftime('%Y-%m-%d %H:%M:%S')}")
