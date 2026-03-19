import json
import tempfile
import unittest
from pathlib import Path

from reporting import build_daily_summary, build_range_summary, date_range_days, write_summary_file


class ReportingTests(unittest.TestCase):
    def test_date_range_days_includes_both_bounds(self):
        self.assertEqual(
            date_range_days("2026-03-10", "2026-03-12"),
            ["2026-03-10", "2026-03-11", "2026-03-12"],
        )

    def test_date_range_days_rejects_reverse_order(self):
        with self.assertRaises(ValueError):
            date_range_days("2026-03-12", "2026-03-10")

    def test_build_daily_summary_counts_events_and_malformed_lines(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            log_path = logs_dir / "2026-03-13.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:00:00",
                                "violation": "slouching",
                                "duration_sec": 4.2,
                                "camera_status": "ok",
                            }
                        ),
                        "{bad json",
                        json.dumps(
                            {
                                "timestamp": "2026-03-13T09:05:00",
                                "violation": "shoulder_tilt",
                                "duration_sec": "5.5",
                                "camera_status": "reconnecting",
                            }
                        ),
                    ]
                ),
                encoding="utf-8",
            )

            summary = build_daily_summary(logs_dir, "2026-03-13")

            self.assertIsNotNone(summary)
            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(summary["malformed_lines"], 1)
            self.assertEqual(summary["camera_status_counts"]["ok"], 1)
            self.assertEqual(summary["camera_status_counts"]["reconnecting"], 1)
            self.assertEqual(summary["violations"]["slouching"]["count"], 1)
            self.assertEqual(summary["violations"]["shoulder_tilt"]["avg_duration_sec"], 5.5)

    def test_build_range_summary_uses_existing_log_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "2026-03-10.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-03-10T10:00:00",
                        "violation": "slouching",
                        "duration_sec": 3,
                        "camera_status": "ok",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (logs_dir / "2026-03-12.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-03-12T10:00:00",
                        "violation": "body_rotation",
                        "duration_sec": 6,
                        "camera_status": "ok",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_range_summary(logs_dir, "2026-03-10", "2026-03-12")

            self.assertIsNotNone(summary)
            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(len(summary["source_log_files"]), 2)
            self.assertEqual(summary["violations"]["body_rotation"]["total_duration_sec"], 6.0)

    def test_write_summary_file_persists_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            summary = {"label": "2026-03-13", "total_events": 1}

            output_path = write_summary_file(logs_dir, "2026-03-13.summary.json", summary)

            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
