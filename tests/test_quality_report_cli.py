import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools import quality_report


class QualityReportCliTests(unittest.TestCase):
    def test_main_returns_error_when_period_is_missing(self):
        with patch("sys.argv", ["quality_report.py"]):
            with io.StringIO() as buf, redirect_stdout(buf):
                exit_code = quality_report.main()
                output = buf.getvalue()

        self.assertEqual(exit_code, 2)
        self.assertIn("Use either --date", output)

    def test_main_returns_error_when_no_logs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sys.argv", ["quality_report.py", "--logs-dir", tmp, "--date", "2026-04-07"]):
                with io.StringIO() as buf, redirect_stdout(buf):
                    exit_code = quality_report.main()
                    output = buf.getvalue()

        self.assertEqual(exit_code, 1)
        self.assertIn("No quality logs found", output)

    def test_main_writes_summary_for_daily_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "2026-04-07.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-07T09:00:00",
                        "violation": "slouching",
                        "posture_state": "bad",
                        "posture_score": 1.3,
                        "tracking_score": 0.93,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.argv", ["quality_report.py", "--logs-dir", tmp, "--date", "2026-04-07"]):
                with io.StringIO() as buf, redirect_stdout(buf):
                    exit_code = quality_report.main()
                    output = buf.getvalue()

            summary_path = logs_dir / "2026-04-07.quality.summary.json"
            self.assertEqual(exit_code, 0)
            self.assertTrue(summary_path.exists())
            self.assertIn("Quality summary written", output)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["posture_state_counts"]["bad"], 1)
            self.assertEqual(summary["avg_posture_score"], 1.3)

    def test_main_prints_summary_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "2026-04-07.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-04-07T09:00:00",
                        "violation": "slouching",
                        "posture_state": "tracking_low",
                        "posture_score": 0.5,
                        "tracking_score": 0.3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.argv", ["quality_report.py", "--logs-dir", tmp, "--date", "2026-04-07", "--print"]):
                with io.StringIO() as buf, redirect_stdout(buf):
                    exit_code = quality_report.main()
                    output = buf.getvalue()

            self.assertEqual(exit_code, 0)
            self.assertIn('"recommendations"', output)


if __name__ == "__main__":
    unittest.main()
