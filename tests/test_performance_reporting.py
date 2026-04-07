import json
import tempfile
import unittest
from pathlib import Path

from performance_reporting import (
    build_daily_perf_summary,
    build_range_perf_summary,
    write_perf_summary_file,
)


class PerformanceReportingTests(unittest.TestCase):
    def test_build_daily_perf_summary_aggregates_fps_switches_and_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            log_path = logs_dir / "2026-03-13.perf.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:00:00",
                                "event": "performance",
                                "provider": "dml",
                                "camera_status": "ok",
                                "fps": 24.5,
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:05:00",
                                "event": "performance",
                                "provider": "cpu",
                                "camera_status": "ok",
                                "fps": 18.0,
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:06:00",
                                "event": "provider_switch",
                                "provider": "cpu",
                                "camera_status": "ok",
                                "previous_provider": "dml",
                                "new_provider": "cpu",
                                "reason": "low_fps_auto_fallback",
                                "fps": 11.2,
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:07:00",
                                "event": "overlay_failure",
                                "provider": "cpu",
                                "camera_status": "ok",
                                "error": "overlay boom",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:08:00",
                                "event": "quality_advice",
                                "provider": "cpu",
                                "camera_status": "ok",
                                "advice": "Tracking stays weak.",
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:09:00",
                                "event": "session_summary",
                                "provider": "cpu",
                                "camera_status": "ok",
                                "session_minutes": 42.0,
                                "present_minutes": 38.0,
                                "away_count": 1,
                                "return_count": 1,
                                "break_reminders": 1,
                                "state_minutes": {"good": 20.0},
                            }
                        ),
                        "{bad json",
                    ]
                ),
                encoding="utf-8",
            )

            summary = build_daily_perf_summary(logs_dir, "2026-03-13")

            self.assertIsNotNone(summary)
            self.assertEqual(summary["total_events"], 6)
            self.assertEqual(summary["malformed_lines"], 1)
            self.assertEqual(summary["event_counts"]["performance"], 2)
            self.assertEqual(summary["fps_by_provider"]["dml"]["avg_fps"], 24.5)
            self.assertEqual(summary["fps_by_provider"]["cpu"]["avg_fps"], 18.0)
            self.assertEqual(summary["provider_switches"][0]["previous_provider"], "dml")
            self.assertEqual(summary["failures"][0]["event"], "overlay_failure")
            self.assertEqual(summary["quality_advice_events"][0]["advice"], "Tracking stays weak.")
            self.assertEqual(summary["session_summaries"][0]["away_count"], 1)

    def test_build_range_perf_summary_uses_existing_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "2026-03-10.perf.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-03-10T10:00:00",
                        "event": "performance",
                        "provider": "dml",
                        "camera_status": "ok",
                        "fps": 20.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (logs_dir / "2026-03-12.perf.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-03-12T10:00:00",
                        "event": "tray_failure",
                        "provider": "cpu",
                        "camera_status": "reconnecting",
                        "error": "tray boom",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_range_perf_summary(logs_dir, "2026-03-10", "2026-03-12")

            self.assertIsNotNone(summary)
            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(len(summary["source_perf_files"]), 2)
            self.assertEqual(summary["camera_status_counts"]["reconnecting"], 1)
            self.assertEqual(summary["provider_counts"]["cpu"], 1)

    def test_write_perf_summary_file_persists_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            summary = {"label": "2026-03-13", "total_events": 3}

            output_path = write_perf_summary_file(logs_dir, "2026-03-13.perf.summary.json", summary)

            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
