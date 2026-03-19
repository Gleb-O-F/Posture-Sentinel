import json
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime as real_datetime
from pathlib import Path
from unittest.mock import patch

from tuning import tune_thresholds


class FixedDateTime:
    @classmethod
    def now(cls):
        return real_datetime(2026, 3, 13, 12, 0, 0)


@dataclass
class ConfigStub:
    slouch_threshold: float = 0.15
    lean_threshold: float = 0.15
    shoulder_tilt_threshold: float = 0.10
    shoulder_width_threshold: float = 0.80


class TuningTests(unittest.TestCase):
    def test_tune_thresholds_requires_at_least_three_days_of_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            for day in ("2026-03-13", "2026-03-12"):
                (logs_dir / f"{day}.jsonl").write_text(
                    json.dumps({"violation": "slouching", "duration_sec": 4, "camera_status": "ok"}) + "\n",
                    encoding="utf-8",
                )

            with patch("tuning.datetime", FixedDateTime):
                report = tune_thresholds(ConfigStub(), logs_dir, lookback_days=7, target_events_per_day=4.0)

            self.assertFalse(report["applied"])
            self.assertEqual(report["reason"], "insufficient_history_days:2")

    def test_tune_thresholds_applies_changes_when_event_volume_is_high(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            events = []
            events.extend({"violation": "slouching", "duration_sec": 4, "camera_status": "ok"} for _ in range(10))
            events.extend({"violation": "leaning_forward", "duration_sec": 5, "camera_status": "ok"} for _ in range(6))
            events.extend({"violation": "body_rotation", "duration_sec": 3, "camera_status": "ok"} for _ in range(4))

            for day in ("2026-03-13", "2026-03-12", "2026-03-11"):
                (logs_dir / f"{day}.jsonl").write_text(
                    "\n".join(json.dumps(item) for item in events) + "\n",
                    encoding="utf-8",
                )

            config = ConfigStub()
            with patch("tuning.datetime", FixedDateTime):
                report = tune_thresholds(config, logs_dir, lookback_days=7, target_events_per_day=2.0)

            self.assertTrue(report["applied"])
            self.assertIn("slouch_threshold", report["changes"])
            self.assertGreater(config.slouch_threshold, 0.15)
            self.assertGreaterEqual(config.shoulder_width_threshold, 0.55)
            self.assertLessEqual(config.shoulder_width_threshold, 0.92)

    def test_tune_thresholds_skips_when_changes_are_not_material(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            daily_events = [{"violation": "slouching", "duration_sec": 4, "camera_status": "ok"} for _ in range(12)]

            for day in ("2026-03-13", "2026-03-12", "2026-03-11"):
                (logs_dir / f"{day}.jsonl").write_text(
                    "\n".join(json.dumps(item) for item in daily_events) + "\n",
                    encoding="utf-8",
                )

            config = ConfigStub()
            with patch("tuning.datetime", FixedDateTime):
                report = tune_thresholds(config, logs_dir, lookback_days=7, target_events_per_day=12.0)

            self.assertFalse(report["applied"])
            self.assertEqual(report["reason"], "no_material_change")
            self.assertEqual(config.slouch_threshold, 0.15)


if __name__ == "__main__":
    unittest.main()
