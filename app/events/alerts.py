"""
Alert engine (Requirement 18, 26).

Generates alerts for fall, unsafe posture, and other configured events,
with a per-(person, alert_type) cooldown to avoid spamming repeated
alerts for the same ongoing situation (Requirement 20).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger("alerts")


@dataclass
class Alert:
    person_id: int
    alert_type: str          # e.g. "fall_detected", "unsafe_bending", "long_inactivity"
    timestamp: float
    message: str
    severity: str = "high"   # low | medium | high
    evidence_path: Optional[str] = None


class AlertEngine:
    def __init__(self, cooldown_seconds: float = 15.0):
        self.cooldown_seconds = cooldown_seconds
        self._last_fired: Dict[Tuple[int, str], float] = {}
        self.active_alerts: List[Alert] = []

    def _on_cooldown(self, person_id: int, alert_type: str, timestamp: float) -> bool:
        key = (person_id, alert_type)
        last = self._last_fired.get(key)
        return last is not None and (timestamp - last) < self.cooldown_seconds

    def fire(self, person_id: int, alert_type: str, timestamp: float, message: str,
              severity: str = "high", evidence_path: Optional[str] = None) -> Optional[Alert]:
        if self._on_cooldown(person_id, alert_type, timestamp):
            return None  # suppressed: avoid repeated alerts (Requirement 20)
        self._last_fired[(person_id, alert_type)] = timestamp
        alert = Alert(person_id=person_id, alert_type=alert_type, timestamp=timestamp,
                       message=message, severity=severity, evidence_path=evidence_path)
        self.active_alerts.append(alert)
        logger.info("ALERT[%s] person=%s: %s", alert_type, person_id, message)
        return alert

    def acknowledge(self, person_id: int, alert_type: str):
        self.active_alerts = [
            a for a in self.active_alerts
            if not (a.person_id == person_id and a.alert_type == alert_type)
        ]
