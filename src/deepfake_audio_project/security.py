import hashlib
import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path):
    hasher = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def verify_file_checksum(path, expected_sha256):
    actual = sha256_file(path)
    return actual.lower() == expected_sha256.lower(), actual


def prediction_entropy(probabilities):
    probs = np.asarray(probabilities, dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def is_ood_prediction(probabilities, confidence_threshold=0.60, entropy_threshold=0.68):
    probs = np.asarray(probabilities, dtype=float)
    max_prob = float(np.max(probs))
    entropy = prediction_entropy(probs)
    ood = (max_prob < confidence_threshold) or (entropy > entropy_threshold)
    return {
        "is_ood": bool(ood),
        "max_probability": max_prob,
        "entropy": entropy,
        "confidence_threshold": confidence_threshold,
        "entropy_threshold": entropy_threshold,
    }


@dataclass
class RiskPolicy:
    low_risk_threshold: float = 0.30
    high_risk_threshold: float = 0.75

    def classify(self, fake_probability, is_ood=False):
        if is_ood:
            return "review"
        if fake_probability >= self.high_risk_threshold:
            return "block"
        if fake_probability >= self.low_risk_threshold:
            return "review"
        return "allow"


class DriftMonitor:
    def __init__(self, window_size=200, alert_std=3.0, min_history=30):
        self.window_size = int(window_size)
        self.alert_std = float(alert_std)
        self.min_history = int(min_history)
        self._history = deque(maxlen=self.window_size)

    def update(self, fake_probability):
        value = float(fake_probability)
        self._history.append(value)
        if len(self._history) < self.min_history:
            return {"drift_alert": False, "reason": "insufficient_history"}

        arr = np.asarray(self._history, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if std < 1e-8:
            z_score = 0.0
            alert = False
        else:
            z_score = float((value - mean) / std)
            alert = abs(z_score) >= self.alert_std

        return {
            "drift_alert": bool(alert),
            "mean_fake_probability": mean,
            "std_fake_probability": std,
            "z_score": z_score,
            "window_size": len(self._history),
        }


class AuditLogger:
    def __init__(self, log_path="outputs/security_audit.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, payload):
        data = dict(payload)
        data["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
        with self.log_path.open("a", encoding="utf-8") as file_obj:
            file_obj.write(json.dumps(data, ensure_ascii=True) + "\n")

