import importlib
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch


class _FakeImageModule:
    @staticmethod
    def new(*args, **kwargs):
        return object()


class _FakeImageDrawModule:
    @staticmethod
    def Draw(_img):
        return SimpleNamespace(ellipse=lambda *args, **kwargs: None)


class _FakePystrayModule:
    @staticmethod
    def Menu(*items):
        return list(items)

    @staticmethod
    def MenuItem(*args, **kwargs):
        return (args, kwargs)

    class Icon:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        def run(self):
            return None


def _install_stubs():
    sys.modules.setdefault("cv2", types.SimpleNamespace())
    sys.modules.setdefault(
        "mediapipe",
        types.SimpleNamespace(solutions=types.SimpleNamespace(pose=types.SimpleNamespace(), drawing_utils=types.SimpleNamespace())),
    )
    sys.modules.setdefault("numpy", types.SimpleNamespace())
    sys.modules.setdefault("onnxruntime", types.SimpleNamespace(InferenceSession=object))
    sys.modules.setdefault("pystray", _FakePystrayModule())
    sys.modules.setdefault("PIL", types.SimpleNamespace(Image=_FakeImageModule, ImageDraw=_FakeImageDrawModule))
    sys.modules.setdefault("PIL.Image", _FakeImageModule)
    sys.modules.setdefault("PIL.ImageDraw", _FakeImageDrawModule)
    sys.modules.setdefault(
        "win32gui",
        types.SimpleNamespace(
            FindWindow=lambda *args, **kwargs: 0,
            GetWindowLong=lambda *args, **kwargs: 0,
            SetWindowLong=lambda *args, **kwargs: None,
            SetWindowPos=lambda *args, **kwargs: None,
        ),
    )
    sys.modules.setdefault(
        "win32con",
        types.SimpleNamespace(GWL_EXSTYLE=0, WS_EX_LAYERED=0, WS_EX_TRANSPARENT=0, HWND_TOPMOST=0, SWP_NOMOVE=0, SWP_NOSIZE=0),
    )
    sys.modules.setdefault("win32api", types.SimpleNamespace())


_install_stubs()
main = importlib.import_module("main")


class MainRuntimeTests(unittest.TestCase):
    def _make_landmarks(
        self,
        *,
        nose_x=0.5,
        nose_y=0.3,
        ls=(0.4, 0.6),
        rs=(0.6, 0.6),
        le=(0.45, 0.32),
        re=(0.55, 0.32),
        tracking=1.0,
    ):
        landmarks = [main.Landmark(0.0, 0.0, 0.0, tracking, tracking) for _ in range(33)]
        landmarks[0] = main.Landmark(nose_x, nose_y, 0.0, tracking, tracking)
        landmarks[7] = main.Landmark(le[0], le[1], 0.0, tracking, tracking)
        landmarks[8] = main.Landmark(re[0], re[1], 0.0, tracking, tracking)
        landmarks[11] = main.Landmark(ls[0], ls[1], 0.0, tracking, tracking)
        landmarks[12] = main.Landmark(rs[0], rs[1], 0.0, tracking, tracking)
        return landmarks

    def test_get_posture_feedback_state_distinguishes_good_and_bad_posture(self):
        app = SimpleNamespace(analyzer=SimpleNamespace(current_violation=None, baseline={"neck_dist": 1.0}))
        bad_eval = main.PostureEvaluation(posture_state="bad", confirmed_violation=("slouching", 3.2))
        pending_eval = main.PostureEvaluation(posture_state="pending")
        tracking_eval = main.PostureEvaluation(posture_state="tracking_low")

        self.assertEqual(main.PostureApp.get_posture_feedback_state(app, bad_eval), "bad")
        self.assertEqual(main.PostureApp.get_posture_feedback_state(app, None), "good")
        self.assertEqual(main.PostureApp.get_posture_feedback_state(app, pending_eval), "pending")
        self.assertEqual(main.PostureApp.get_posture_feedback_state(app, tracking_eval), "tracking_low")

        app.analyzer.current_violation = "slouching"
        self.assertEqual(main.PostureApp.get_posture_feedback_state(app, None), "pending")

        app.analyzer.current_violation = None
        app.analyzer.baseline = None
        self.assertEqual(main.PostureApp.get_posture_feedback_state(app, None), "uncalibrated")

    def test_config_load_invalidates_legacy_baseline_shape(self):
        with patch.object(main.Path, "exists", return_value=True), patch.object(
            main, "open", create=True
        ) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                '{"baseline":{"shoulder_y_norm":1.33,"ear_dist_ratio":0.41,"shoulder_width_base":0.65,"shoulder_tilt_base":0.05}}'
            )
            config = main.Config.load("config.yaml")

        self.assertIsNone(config.baseline)

    def test_posture_analyzer_treats_calibrated_pose_as_good(self):
        analyzer = main.PostureAnalyzer()
        config = main.Config()
        landmarks = self._make_landmarks()

        analyzer.calibrate(landmarks, config)
        evaluation = analyzer.evaluate(landmarks, config)

        self.assertIsNone(evaluation.confirmed_violation)
        self.assertEqual(evaluation.posture_state, "good")
        self.assertIsNone(analyzer.current_violation)

    def test_posture_analyzer_reports_slouch_after_timeout(self):
        analyzer = main.PostureAnalyzer()
        config = main.Config(slouch_threshold=0.08, violation_timeout_sec=3.0)
        good_landmarks = self._make_landmarks()
        slouch_landmarks = self._make_landmarks(nose_y=0.42, le=(0.45, 0.43), re=(0.55, 0.43))

        analyzer.calibrate(good_landmarks, config)
        with patch.object(main.time, "time", side_effect=[10.0, 14.2]):
            self.assertIsNone(analyzer.check(slouch_landmarks, config))
            violation = analyzer.check(slouch_landmarks, config)

        self.assertEqual(violation[0], "slouching")

    def test_calibration_requires_good_tracking_quality(self):
        analyzer = main.PostureAnalyzer()
        config = main.Config(min_calibration_tracking_confidence=0.7)
        weak_landmarks = self._make_landmarks(tracking=0.35)

        baseline = analyzer.calibrate(weak_landmarks, config)

        self.assertIsNone(baseline)
        self.assertEqual(analyzer.calibration_message, "Tracking quality is too low for calibration.")

    def test_posture_analyzer_marks_low_tracking_as_separate_state(self):
        analyzer = main.PostureAnalyzer()
        config = main.Config(min_tracking_confidence=0.6)
        good_landmarks = self._make_landmarks()
        weak_landmarks = self._make_landmarks(tracking=0.2)

        analyzer.calibrate(good_landmarks, config)
        evaluation = analyzer.evaluate(weak_landmarks, config)

        self.assertEqual(evaluation.posture_state, "tracking_low")
        self.assertIsNone(evaluation.confirmed_violation)

    def test_update_quality_advice_warns_about_low_tracking(self):
        app = SimpleNamespace(
            config=main.Config(quality_advice_enabled=True, min_calibration_tracking_confidence=0.7),
            quality_advice=None,
            low_tracking_streak=19,
            pending_posture_streak=0,
        )
        evaluation = main.PostureEvaluation(posture_state="tracking_low", tracking_score=0.2)

        main.PostureApp.update_quality_advice(app, evaluation)

        self.assertIn("Tracking stays weak", app.quality_advice)

    def test_update_quality_advice_suggests_recalibration_for_borderline_posture(self):
        app = SimpleNamespace(
            config=main.Config(quality_advice_enabled=True, min_calibration_tracking_confidence=0.7),
            quality_advice=None,
            low_tracking_streak=0,
            pending_posture_streak=29,
        )
        evaluation = main.PostureEvaluation(posture_state="pending", tracking_score=0.88, posture_score=0.7)

        main.PostureApp.update_quality_advice(app, evaluation)

        self.assertIn("borderline", app.quality_advice)

    def test_adjust_posture_sensitivity_clamps_and_saves(self):
        saves = []
        app = SimpleNamespace(config=main.Config(posture_sensitivity=1.45))
        app.config.save = lambda: saves.append(app.config.posture_sensitivity)

        main.PostureApp.adjust_posture_sensitivity(app, 0.2)

        self.assertEqual(app.config.posture_sensitivity, 1.5)
        self.assertEqual(len(saves), 1)

    def test_update_quality_advice_logs_new_advice_once(self):
        logged = []
        app = SimpleNamespace(
            config=main.Config(quality_advice_enabled=True, min_calibration_tracking_confidence=0.7),
            quality_advice=None,
            last_logged_quality_advice=None,
            low_tracking_streak=19,
            pending_posture_streak=0,
            log_runtime_event=lambda event, **payload: logged.append((event, payload)),
        )
        evaluation = main.PostureEvaluation(posture_state="tracking_low", tracking_score=0.2)

        main.PostureApp.update_quality_advice(app, evaluation)
        main.PostureApp.update_quality_advice(app, evaluation)

        self.assertEqual(len(logged), 1)
        self.assertEqual(logged[0][0], "quality_advice")

    def test_export_quality_summary_returns_written_file(self):
        with patch.object(main, "build_daily_quality_summary", return_value={"label": "2026-04-07"}) as build_mock, patch.object(
            main, "write_quality_summary_file", return_value=main.Path("logs/2026-04-07.quality.summary.json")
        ) as write_mock:
            app = SimpleNamespace(logs_dir=main.Path("logs"))

            result = main.PostureApp.export_quality_summary(app, "2026-04-07")

        self.assertEqual(str(result), "logs\\2026-04-07.quality.summary.json")
        build_mock.assert_called_once()
        write_mock.assert_called_once()

    def test_run_overlay_disables_overlay_after_failure(self):
        app = SimpleNamespace(config=SimpleNamespace(overlay_color="#FFFFFF"), overlay=None, overlay_available=True)

        class BrokenOverlay:
            def __init__(self, color):
                self.color = color

            def run(self):
                raise RuntimeError("overlay boom")

        with patch.object(main, "Overlay", BrokenOverlay):
            main.PostureApp.run_overlay(app)

        self.assertIsNone(app.overlay)
        self.assertFalse(app.overlay_available)

    def test_run_tray_disables_tray_after_failure(self):
        app = SimpleNamespace(
            toggle_preview=lambda *args, **kwargs: None,
            toggle_auto_start=lambda *args, **kwargs: None,
            toggle_auto_tune=lambda *args, **kwargs: None,
            toggle_quality_advice=lambda *args, **kwargs: None,
            increase_posture_sensitivity=lambda *args, **kwargs: None,
            decrease_posture_sensitivity=lambda *args, **kwargs: None,
            increase_good_posture_threshold=lambda *args, **kwargs: None,
            decrease_good_posture_threshold=lambda *args, **kwargs: None,
            request_calibration=lambda *args, **kwargs: None,
            auto_tune_thresholds=lambda *args, **kwargs: None,
            export_today_summary=lambda *args, **kwargs: None,
            export_today_quality_summary=lambda *args, **kwargs: None,
            on_exit=lambda *args, **kwargs: None,
            create_icon_image=lambda *args, **kwargs: object(),
            show_preview=True,
            config=SimpleNamespace(auto_start=False, auto_tune_enabled=True, quality_advice_enabled=True),
            icon=None,
            tray_available=True,
        )

        class BrokenPystray:
            @staticmethod
            def Menu(*items):
                return list(items)

            @staticmethod
            def MenuItem(*args, **kwargs):
                return (args, kwargs)

            class Icon:
                def __init__(self, *args, **kwargs):
                    raise RuntimeError("tray boom")

        with patch.object(main, "pystray", BrokenPystray):
            started = main.PostureApp.run_tray(app)

        self.assertFalse(started)
        self.assertIsNone(app.icon)
        self.assertFalse(app.tray_available)

    def test_run_tray_returns_true_when_icon_runs(self):
        app = SimpleNamespace(
            toggle_preview=lambda *args, **kwargs: None,
            toggle_auto_start=lambda *args, **kwargs: None,
            toggle_auto_tune=lambda *args, **kwargs: None,
            toggle_quality_advice=lambda *args, **kwargs: None,
            increase_posture_sensitivity=lambda *args, **kwargs: None,
            decrease_posture_sensitivity=lambda *args, **kwargs: None,
            increase_good_posture_threshold=lambda *args, **kwargs: None,
            decrease_good_posture_threshold=lambda *args, **kwargs: None,
            request_calibration=lambda *args, **kwargs: None,
            auto_tune_thresholds=lambda *args, **kwargs: None,
            export_today_summary=lambda *args, **kwargs: None,
            export_today_quality_summary=lambda *args, **kwargs: None,
            on_exit=lambda *args, **kwargs: None,
            create_icon_image=lambda *args, **kwargs: object(),
            show_preview=False,
            config=SimpleNamespace(auto_start=True, auto_tune_enabled=False, quality_advice_enabled=True),
            icon=None,
            tray_available=True,
        )

        class RecordingIcon:
            def __init__(self, *args, **kwargs):
                self.ran = False

            def run(self):
                self.ran = True

        class GoodPystray:
            @staticmethod
            def Menu(*items):
                return list(items)

            @staticmethod
            def MenuItem(*args, **kwargs):
                return (args, kwargs)

            Icon = RecordingIcon

        with patch.object(main, "pystray", GoodPystray):
            started = main.PostureApp.run_tray(app)

        self.assertTrue(started)
        self.assertIsInstance(app.icon, RecordingIcon)
        self.assertTrue(app.icon.ran)
        self.assertTrue(app.tray_available)


if __name__ == "__main__":
    unittest.main()
