import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from tools import perf_report


class PerfReportCliTests(unittest.TestCase):
    def test_main_returns_error_when_period_is_missing(self):
        with patch("sys.argv", ["perf_report.py"]):
            with io.StringIO() as buf, redirect_stdout(buf):
                exit_code = perf_report.main()
                output = buf.getvalue()

        self.assertEqual(exit_code, 2)
        self.assertIn("Use either --date", output)

    def test_main_returns_error_when_no_logs_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("sys.argv", ["perf_report.py", "--logs-dir", tmp, "--date", "2026-03-13"]):
                with io.StringIO() as buf, redirect_stdout(buf):
                    exit_code = perf_report.main()
                    output = buf.getvalue()

        self.assertEqual(exit_code, 1)
        self.assertIn("No performance logs found", output)

    def test_main_writes_summary_for_daily_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            logs_dir = Path(tmp)
            (logs_dir / "2026-03-13.perf.jsonl").write_text(
                json.dumps(
                    {
                        "timestamp": "2026-03-13T09:00:00",
                        "event": "performance",
                        "provider": "dml",
                        "camera_status": "ok",
                        "fps": 25.0,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with patch("sys.argv", ["perf_report.py", "--logs-dir", tmp, "--date", "2026-03-13"]):
                with io.StringIO() as buf, redirect_stdout(buf):
                    exit_code = perf_report.main()
                    output = buf.getvalue()

            summary_path = logs_dir / "2026-03-13.perf.summary.json"
            self.assertEqual(exit_code, 0)
            self.assertTrue(summary_path.exists())
            self.assertIn("Performance summary written", output)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["event_counts"]["performance"], 1)
            self.assertEqual(summary["fps_by_provider"]["dml"]["avg_fps"], 25.0)


if __name__ == "__main__":
    unittest.main()
