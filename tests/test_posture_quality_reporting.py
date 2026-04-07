import json
import tempfile
import unittest
from pathlib import Path

from posture_quality_reporting import (
    build_daily_quality_summary,
    build_range_quality_summary,
    write_quality_summary_file,
)


class PostureQualityReportingTests(unittest.TestCase):
    def test_build_daily_quality_summary_aggregates_scores_and_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            log_path = logs_dir / "2026-04-07.jsonl"
            log_path.write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-04-07T09:00:00",
                                "violation": "slouching",
                                "posture_state": "bad",
                                "posture_score": 1.24,
                                "tracking_score": 0.91,
                            }
                        ),
                        json.dumps(
                            {
                                "timestamp": "2026-04-07T10:00:00",
                                "violation": "leaning_forward",
                                "posture_state": "pending",
                                "posture_score": 0.82,
                                "tracking_score": 0.74,
                            }
                        ),
                        "{bad json",
                    ]
                ),
                encoding="utf-8",
            )

            summary = build_daily_quality_summary(logs_dir, "2026-04-07")

            self.assertIsNotNone(summary)
            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(summary["malformed_lines"], 1)
            self.assertEqual(summary["posture_state_counts"]["bad"], 1)
            self.assertEqual(summary["posture_state_counts"]["pending"], 1)
            self.assertEqual(summary["violation_quality"]["slouching"]["avg_posture_score"], 1.24)
            self.assertEqual(summary["avg_tracking_score"], 0.825)
            self.assertTrue(summary["recommendations"])

    def test_build_range_quality_summary_uses_existing_files_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "2026-04-05.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-05T09:00:00",
                        "violation": "slouching",
                        "posture_state": "bad",
                        "posture_score": 1.1,
                        "tracking_score": 0.8,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (logs_dir / "2026-04-07.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-07T09:00:00",
                        "violation": "shoulder_tilt",
                        "posture_state": "bad",
                        "posture_score": 1.4,
                        "tracking_score": 0.88,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_range_quality_summary(logs_dir, "2026-04-05", "2026-04-07")

            self.assertIsNotNone(summary)
            self.assertEqual(summary["total_events"], 2)
            self.assertEqual(len(summary["source_log_files"]), 2)
            self.assertEqual(summary["max_posture_score"], 1.4)
            self.assertEqual(len(summary["daily_trends"]), 2)
            self.assertEqual(summary["daily_trends"][0]["day"], "2026-04-05")

    def test_build_quality_summary_recommends_tracking_improvements(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "2026-04-07.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps(
                            {
                                "timestamp": "2026-04-07T09:00:00",
                                "violation": "slouching",
                                "posture_state": "tracking_low",
                                "posture_score": 0.5,
                                "tracking_score": 0.3,
                            }
                        )
                        for _ in range(4)
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_daily_quality_summary(logs_dir, "2026-04-07")

            self.assertIsNotNone(summary)
            self.assertIn("Tracking quality is frequently low", summary["recommendations"][0])

    def test_write_quality_summary_file_persists_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            summary = {"label": "2026-04-07", "total_events": 3}

            output_path = write_quality_summary_file(logs_dir, "2026-04-07.quality.summary.json", summary)

            self.assertTrue(output_path.exists())
            self.assertEqual(json.loads(output_path.read_text(encoding="utf-8")), summary)


if __name__ == "__main__":
    unittest.main()
