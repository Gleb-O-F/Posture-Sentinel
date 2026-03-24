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
    def test_config_load_normalizes_legacy_baseline_shape(self):
        with patch.object(main.Path, "exists", return_value=True), patch.object(
            main, "open", create=True
        ) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = (
                '{"baseline":{"shoulder_y_norm":1.33,"ear_dist_ratio":0.41,"shoulder_width_base":0.65,"shoulder_tilt_base":0.05}}'
            )
            config = main.Config.load("config.yaml")

        self.assertEqual(
            config.baseline,
            {
                "neck_dist": 1.33,
                "ear_ratio": 0.41,
                "width": 0.65,
                "tilt": 0.05,
            },
        )

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
            request_calibration=lambda *args, **kwargs: None,
            auto_tune_thresholds=lambda *args, **kwargs: None,
            export_today_summary=lambda *args, **kwargs: None,
            on_exit=lambda *args, **kwargs: None,
            create_icon_image=lambda *args, **kwargs: object(),
            show_preview=True,
            config=SimpleNamespace(auto_start=False, auto_tune_enabled=True),
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
            request_calibration=lambda *args, **kwargs: None,
            auto_tune_thresholds=lambda *args, **kwargs: None,
            export_today_summary=lambda *args, **kwargs: None,
            on_exit=lambda *args, **kwargs: None,
            create_icon_image=lambda *args, **kwargs: object(),
            show_preview=False,
            config=SimpleNamespace(auto_start=True, auto_tune_enabled=False),
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
