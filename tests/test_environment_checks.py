import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import environment_checks


class EnvironmentChecksTests(unittest.TestCase):
    def test_check_models_detects_missing_required_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            report = environment_checks.check_models(models_dir)

        self.assertFalse(report["ok"])
        self.assertIn("pose_landmarks_detector_full.onnx", report["missing"])

    def test_check_modules_reports_missing_packages(self):
        with patch("importlib.util.find_spec", return_value=None):
            report = environment_checks.check_modules(platform="win32")

        self.assertFalse(report["ok"])
        self.assertIn("cv2", report["missing"])
        self.assertIn("win32gui", report["missing"])

    def test_build_environment_report_is_ok_when_everything_is_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "pose_landmarks_detector_full.onnx").write_text("ok", encoding="utf-8")

            with patch.object(environment_checks, "get_python_info", return_value={"supported": True}):
                with patch.object(environment_checks, "check_modules", return_value={"ok": True, "available": {}, "missing": {}}):
                    report = environment_checks.build_environment_report(models_dir=models_dir, include_camera=False)

        self.assertTrue(report["ok"])
        self.assertEqual(report["camera"]["checked"], False)

    def test_build_environment_report_includes_camera_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            models_dir = Path(tmp)
            (models_dir / "pose_landmarks_detector_full.onnx").write_text("ok", encoding="utf-8")

            with patch.object(environment_checks, "get_python_info", return_value={"supported": True}):
                with patch.object(environment_checks, "check_modules", return_value={"ok": True, "available": {}, "missing": {}}):
                    with patch.object(environment_checks, "check_camera", return_value={"checked": True, "ok": False, "reason": "open_failed"}):
                        report = environment_checks.build_environment_report(models_dir=models_dir, include_camera=True)

        self.assertFalse(report["ok"])
        self.assertEqual(report["camera"]["reason"], "open_failed")


if __name__ == "__main__":
    unittest.main()
