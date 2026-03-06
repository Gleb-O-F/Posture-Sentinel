import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
from typing import Optional, Dict
from pathlib import Path
import time
import math
import json
import threading
import pystray
from PIL import Image, ImageDraw
import win32gui
import win32con
import win32api
import tkinter as tk
import os
import sys
import ctypes
import argparse
from datetime import datetime
from reporting import build_daily_summary, write_summary_file
from tuning import tune_thresholds

# WinAPI for Acrylic/Blur effect
class ACCENTPOLICY(ctypes.Structure):
    _fields_ = [
        ("AccentState", ctypes.c_uint),
        ("AccentFlags", ctypes.c_uint),
        ("GradientColor", ctypes.c_uint),
        ("AnimationId", ctypes.c_uint),
    ]

class WINDOWCOMPOSITIONATTRIBUTEDATA(ctypes.Structure):
    _fields_ = [
        ("Attribute", ctypes.c_int),
        ("Data", ctypes.POINTER(ACCENTPOLICY)),
        ("SizeOfData", ctypes.c_size_t),
    ]

def set_blur(hwnd):
    accent = ACCENTPOLICY()
    accent.AccentState = 4 # Acrylic effect
    accent.GradientColor = 0x50FFFFFF # Semi-transparent white
    data = WINDOWCOMPOSITIONATTRIBUTEDATA()
    data.Attribute = 19
    data.SizeOfData = ctypes.sizeof(accent)
    data.Data = ctypes.pointer(accent)
    ctypes.windll.user32.SetWindowCompositionAttribute(hwnd, ctypes.pointer(data))

@dataclass
class Config:
    camera_id: int = 0
    slouch_threshold: float = 0.15     # Порог сжатия шеи (относительно ширины плеч)
    lean_threshold: float = 0.15       # Порог наклона вперед (по ушам)
    shoulder_tilt_threshold: float = 0.10 # Порог перекоса плеч
    shoulder_width_threshold: float = 0.8  # Порог разворота
    violation_timeout_sec: float = 3.0
    show_preview: bool = True
    overlay_color: str = "#FFFFFF"
    baseline: Optional[Dict[str, float]] = None
    camera_reconnect_interval_sec: float = 1.0
    camera_read_fail_limit: int = 15
    performance_window_sec: float = 8.0
    min_fps_before_cpu_fallback: float = 12.0
    enable_auto_cpu_fallback: bool = True
    auto_start: bool = False
    auto_tune_enabled: bool = True
    auto_tune_lookback_days: int = 7
    auto_tune_target_events_per_day: float = 12.0

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        if Path(path).exists():
            with open(path) as f:
                try:
                    data = json.load(f)
                    valid_keys = cls.__dataclass_fields__.keys()
                    filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                    return cls(**filtered_data)
                except Exception:
                    return cls()
        return cls()

    def save(self, path: str = "config.yaml"):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

class Overlay:
    def __init__(self, color="#FFFFFF"):
        self.root = None
        self.color = color
        self.target_alpha = 0.0
        self.current_alpha = 0.0
        self.running = True

    def set_alpha(self, alpha):
        self.target_alpha = max(0.0, min(0.85, alpha))

    def update(self):
        if not self.running or not self.root:
            return
        if abs(self.current_alpha - self.target_alpha) > 0.005:
            step = 0.015 if self.target_alpha > self.current_alpha else 0.04
            self.current_alpha = min(self.target_alpha, self.current_alpha + step) if self.current_alpha < self.target_alpha else max(self.target_alpha, self.current_alpha - step)
            try:
                self.root.attributes("-alpha", self.current_alpha)
            except tk.TclError:
                pass
        self.root.after(20, self.update)

    def run(self):
        self.root = tk.Tk()
        self.root.title("PostureSentinelOverlay")
        self.root.attributes("-topmost", True, "-alpha", 0.0, "-fullscreen", True)
        self.root.config(bg=self.color)
        self.root.overrideredirect(True)
        self.root.update()
        hwnd = win32gui.FindWindow(None, "PostureSentinelOverlay")
        if hwnd:
            ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
            win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE, ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)
            set_blur(hwnd)
            win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
        self.update()
        self.root.mainloop()

    def stop(self):
        self.running = False
        if self.root:
            try: self.root.after(0, self.root.destroy)
            except: pass

class Landmark:
    def __init__(self, x, y, z, visibility=None, presence=None):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence

class ONNXPoseDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.active_provider = "uninitialized"
        self._init_session(prefer_dml=True)

    def _init_session(self, prefer_dml: bool):
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider'] if prefer_dml else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            active_providers = self.session.get_providers()
            self.active_provider = active_providers[0] if active_providers else "unknown"
            print(f"ONNX Session initialized. Active providers: {active_providers}")
        except Exception as e:
            if prefer_dml:
                print(f"Warning: Failed to initialize ONNX with DML, falling back to CPU: {e}")
            else:
                print(f"Warning: Failed to initialize ONNX session: {e}")
            self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
            self.active_provider = "CPUExecutionProvider"

    def provider_short(self):
        if self.active_provider == "DmlExecutionProvider":
            return "dml"
        if self.active_provider == "CPUExecutionProvider":
            return "cpu"
        return self.active_provider.lower()

    def fallback_to_cpu(self) -> bool:
        if self.active_provider == "CPUExecutionProvider":
            return False
        try:
            self._init_session(prefer_dml=False)
            return self.active_provider == "CPUExecutionProvider"
        except Exception as e:
            print(f"Warning: Failed to fallback to CPU provider: {e}")
            return False

    def process(self, frame):
        # Preprocessing: 256x256, normalize to [0, 1], NHWC format
        img = cv2.resize(frame, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img[np.newaxis, ...] # Add batch dimension: (1, 256, 256, 3)

        # Inference
        # The model usually has multiple outputs; we need the landmark one (Identity)
        input_name = self.session.get_inputs()[0].name
        outputs = self.session.run(None, {input_name: img})

        # Postprocessing: Extract 33 points * 5 parameters (x, y, z, visibility, presence)
        # Based on the plan, landmarks are in the first 165 values of the first output
        landmarks_raw = outputs[0].flatten()

        if len(landmarks_raw) < 165:
            return type('Result', (), {'pose_landmarks': None})()

        landmarks = []
        for i in range(0, 165, 5):
            # Normalize x, y by 256.0 as they are in pixel coordinates in the tensor
            x = landmarks_raw[i] / 256.0
            y = landmarks_raw[i+1] / 256.0
            z = landmarks_raw[i+2] / 256.0
            vis = landmarks_raw[i+3]
            pres = landmarks_raw[i+4]
            landmarks.append(Landmark(x, y, z, vis, pres))

        # Wrap for compatibility with existing code
        pose_landmarks = type('PoseLandmarks', (), {'landmark': landmarks})()
        return type('Result', (), {'pose_landmarks': pose_landmarks})()

class PostureAnalyzer:
    def __init__(self, baseline=None):
        self.baseline = baseline
        self.violation_start: Optional[float] = None
        self.current_violation: Optional[str] = None

    def calibrate(self, landmarks):
        ls, rs, le, re, nose = landmarks[11], landmarks[12], landmarks[7], landmarks[8], landmarks[0]
        width = abs(ls.x - rs.x)
        if width <= 0: return None

        # Расстояние от носа до линии плеч по вертикали, нормированное шириной плеч
        neck_dist = ((ls.y + rs.y) / 2 - nose.y) / width

        self.baseline = {
            "neck_dist": neck_dist,
            "ear_ratio": abs(le.x - re.x) / width,
            "tilt": (ls.y - rs.y) / width,
            "width": width
        }
        return self.baseline

    def check(self, landmarks, config: Config) -> Optional[tuple[str, float]]:
        if not self.baseline or "neck_dist" not in self.baseline: return None

        ls, rs, le, re, nose = landmarks[11], landmarks[12], landmarks[7], landmarks[8], landmarks[0]
        curr_width = abs(ls.x - rs.x)
        if curr_width <= 0.05: return None

        curr_neck_dist = ((ls.y + rs.y) / 2 - nose.y) / curr_width
        curr_ear_ratio = abs(le.x - re.x) / curr_width
        curr_tilt = (ls.y - rs.y) / curr_width

        violation = None
        # 1. Сутулость: когда плечи поднимаются к ушам или голова падает (дистанция сокращается)
        if self.baseline["neck_dist"] - curr_neck_dist > config.slouch_threshold:
            violation = "slouching"
        # 2. Наклон вперед: голова становится больше относительно плеч
        elif curr_ear_ratio / self.baseline["ear_ratio"] > 1.0 + config.lean_threshold:
            violation = "leaning_forward"
        # 3. Перекос плеч
        elif abs(curr_tilt - self.baseline["tilt"]) > config.shoulder_tilt_threshold:
            violation = "shoulder_tilt"
        # 4. Разворот корпуса
        elif curr_width / self.baseline["width"] < config.shoulder_width_threshold:
            violation = "body_rotation"

        if violation:
            if self.current_violation == violation and self.violation_start:
                elapsed = time.time() - self.violation_start
                if elapsed >= config.violation_timeout_sec: return (violation, elapsed)
            else:
                self.current_violation = violation
                self.violation_start = time.time()
        else:
            self.current_violation = None
            self.violation_start = None
        return None

def apply_blur_effect(frame, intensity: float = 0.5):
    h, w = frame.shape[:2]
    blur_amount = int(51 * intensity)
    if blur_amount % 2 == 0: blur_amount += 1
    return cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)

class PostureApp:
    def __init__(self, headless: bool = False, no_tray: bool = False):
        self.config = Config.load()
        self.analyzer = PostureAnalyzer(self.config.baseline)
        self.headless = headless
        self.no_tray = no_tray
        initial_preview = False if self.headless else self.config.show_preview
        self.running, self.show_preview, self.calibrate_requested = True, initial_preview, False
        self.icon, self.overlay, self.blur_intensity = None, None, 0.0
        self.camera_status = "init"
        self.logs_dir = Path("logs")
        self.last_logged_violation: Optional[str] = None

        model_path = os.path.join("models", "pose_landmarks_detector_full.onnx")
        self.detector = ONNXPoseDetector(model_path)
        self.sync_auto_start_with_config()

    def set_camera_status(self, status: str):
        if status != self.camera_status:
            self.camera_status = status
            print(f"Camera status: {status}")

    def update_tray_status(self, posture_color: str):
        if self.camera_status == "ok":
            tray_color = posture_color
        elif self.camera_status == "reconnecting":
            tray_color = "orange"
        else:
            tray_color = "gray"

        if self.icon:
            self.icon.icon = self.create_icon_image(tray_color)
            provider = self.detector.provider_short().upper() if self.detector else "N/A"
            self.icon.title = f"Posture Sentinel ({self.camera_status}, {provider})"

    def log_violation(self, violation_type: str, duration_sec: float):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now()
        log_file = self.logs_dir / f"{ts.strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "timestamp": ts.isoformat(timespec="seconds"),
            "violation": violation_type,
            "duration_sec": round(duration_sec, 2),
            "camera_status": self.camera_status,
        }
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def export_daily_summary(self, day: Optional[str] = None) -> Optional[Path]:
        target_day = day or datetime.now().strftime("%Y-%m-%d")
        summary = build_daily_summary(self.logs_dir, target_day)
        if summary is None:
            print(f"Summary skipped: no log file for {target_day}")
            return None

        summary_file = write_summary_file(self.logs_dir, f"{target_day}.summary.json", summary)
        print(f"Daily summary exported: {summary_file}")
        return summary_file

    def export_today_summary(self, icon=None, item=None):
        self.export_daily_summary()

    def auto_tune_thresholds(self, icon=None, item=None):
        if not self.config.auto_tune_enabled:
            print("Auto-tune is disabled in config")
            return

        report = tune_thresholds(
            config_obj=self.config,
            logs_dir=self.logs_dir,
            lookback_days=self.config.auto_tune_lookback_days,
            target_events_per_day=self.config.auto_tune_target_events_per_day,
        )
        if report.get("applied"):
            self.config.save()
            day = datetime.now().strftime("%Y-%m-%d")
            report_file = self.logs_dir / f"{day}.autotune.json"
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            print(f"Auto-tune applied and saved: {report_file}")
            print(f"Threshold changes: {report.get('changes', {})}")
        else:
            print(f"Auto-tune skipped: {report.get('reason', 'unknown')}")

    def create_icon_image(self, color="green"):
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((4, 4, 60, 60), fill=color)
        return img

    def get_startup_launcher_path(self) -> Optional[Path]:
        appdata = os.getenv("APPDATA")
        if not appdata:
            return None
        startup_dir = Path(appdata) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
        return startup_dir / "PostureSentinel.bat"

    def is_auto_start_enabled(self) -> bool:
        launcher = self.get_startup_launcher_path()
        return bool(launcher and launcher.exists())

    def set_auto_start(self, enabled: bool) -> bool:
        launcher = self.get_startup_launcher_path()
        if launcher is None:
            print("Auto-start unavailable: APPDATA not found")
            return False

        try:
            if enabled:
                launcher.parent.mkdir(parents=True, exist_ok=True)
                script_path = Path(__file__).resolve()
                python_path = Path(sys.executable).resolve()
                pythonw_path = python_path.with_name("pythonw.exe")
                launcher_python = pythonw_path if pythonw_path.exists() else python_path
                content = (
                    "@echo off\n"
                    f'cd /d "{script_path.parent}"\n'
                    f'"{launcher_python}" "{script_path}" --headless\n'
                )
                with open(launcher, "w", encoding="utf-8") as f:
                    f.write(content)
                print(f"Auto-start enabled: {launcher}")
            else:
                if launcher.exists():
                    launcher.unlink()
                print("Auto-start disabled")
            return True
        except Exception as e:
            print(f"Auto-start update failed: {e}")
            return False

    def sync_auto_start_with_config(self):
        current = self.is_auto_start_enabled()
        desired = bool(self.config.auto_start)
        if current == desired:
            return
        if self.set_auto_start(desired):
            self.config.auto_start = desired
            self.config.save()

    def on_exit(self, icon, item):
        self.running = False
        if self.overlay: self.overlay.stop()
        self.export_daily_summary()
        icon.stop()
        os._exit(0)

    def toggle_preview(self, icon, item):
        if self.headless:
            return
        self.show_preview = not self.show_preview
        self.config.show_preview = self.show_preview
        self.config.save()

    def request_calibration(self, icon=None, item=None):
        self.calibrate_requested = True

    def toggle_auto_start(self, icon=None, item=None):
        desired = not self.config.auto_start
        if self.set_auto_start(desired):
            self.config.auto_start = desired
            self.config.save()

    def toggle_auto_tune(self, icon=None, item=None):
        self.config.auto_tune_enabled = not self.config.auto_tune_enabled
        self.config.save()
        print(f"Auto-tune enabled: {self.config.auto_tune_enabled}")

    def run_overlay(self):
        self.overlay = Overlay(self.config.overlay_color)
        self.overlay.run()

    def run_tray(self):
        menu = pystray.Menu(pystray.MenuItem("Show/Hide Preview", self.toggle_preview, checked=lambda item: self.show_preview),
                            pystray.MenuItem("Auto Start", self.toggle_auto_start, checked=lambda item: self.config.auto_start),
                            pystray.MenuItem("Auto Tune", self.toggle_auto_tune, checked=lambda item: self.config.auto_tune_enabled),
                            pystray.MenuItem("Calibrate", self.request_calibration),
                            pystray.MenuItem("Auto-tune Thresholds", self.auto_tune_thresholds),
                            pystray.MenuItem("Export Daily Summary", self.export_today_summary),
                            pystray.MenuItem("Exit", self.on_exit))
        self.icon = pystray.Icon("PostureSentinel", self.create_icon_image(), "Posture Sentinel", menu)
        self.icon.run()

    def run_cv(self):
        mp_pose = mp.solutions.pose
        drawing = mp.solutions.drawing_utils
        cap = None
        consecutive_read_failures = 0
        last_reconnect_attempt = 0.0
        perf_window_start = time.time()
        perf_frame_count = 0
        current_fps = 0.0
        cpu_fallback_done = False
        while self.running:
            if cap is None or not cap.isOpened():
                now = time.time()
                if now - last_reconnect_attempt >= self.config.camera_reconnect_interval_sec:
                    last_reconnect_attempt = now
                    if cap is not None:
                        cap.release()
                    cap = cv2.VideoCapture(self.config.camera_id)
                    if cap.isOpened():
                        self.set_camera_status("ok")
                        consecutive_read_failures = 0
                    else:
                        cap.release()
                        cap = None
                        self.set_camera_status("reconnecting")

                self.update_tray_status("gray")
                if self.show_preview:
                    standby = np.zeros((360, 640, 3), dtype=np.uint8)
                    cv2.putText(standby, "Camera unavailable", (150, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    cv2.putText(standby, "Trying to reconnect...", (145, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 180, 255), 2)
                    cv2.imshow("Posture Sentinel", standby)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.toggle_preview(None, None)
                    elif key == ord('c'):
                        self.calibrate_requested = True
                else:
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                time.sleep(0.1)
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= self.config.camera_read_fail_limit:
                    self.set_camera_status("lost")
                    cap.release()
                    cap = None
                self.update_tray_status("gray")
                time.sleep(0.05)
                continue

            consecutive_read_failures = 0
            self.set_camera_status("ok")
            frame = cv2.flip(frame, 1)
            results = self.detector.process(frame)
            perf_frame_count += 1
            now = time.time()
            elapsed_perf = now - perf_window_start
            if elapsed_perf >= self.config.performance_window_sec:
                current_fps = perf_frame_count / max(elapsed_perf, 1e-6)
                print(f"Performance: {current_fps:.1f} FPS | provider={self.detector.provider_short()}")
                if (
                    self.config.enable_auto_cpu_fallback
                    and not cpu_fallback_done
                    and self.detector.active_provider == "DmlExecutionProvider"
                    and current_fps < self.config.min_fps_before_cpu_fallback
                ):
                    if self.detector.fallback_to_cpu():
                        cpu_fallback_done = True
                        print(
                            f"Auto fallback: switched to CPU due to low FPS "
                            f"({current_fps:.1f} < {self.config.min_fps_before_cpu_fallback:.1f})"
                        )
                perf_window_start = now
                perf_frame_count = 0
            status_color = "green"
            violation = None
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if self.calibrate_requested:
                    self.config.baseline = self.analyzer.calibrate(landmarks)
                    self.config.save(); self.calibrate_requested = False
                    print("Calibrated with new nose-to-shoulder model!")
                violation = self.analyzer.check(landmarks, self.config)
                if violation:
                    status_color = "red"
                    self.blur_intensity = min(1.0, self.blur_intensity + 0.08)
                    if self.last_logged_violation != violation[0]:
                        self.log_violation(violation[0], violation[1])
                        self.last_logged_violation = violation[0]
                elif self.analyzer.current_violation:
                    status_color = "orange"
                else:
                    self.blur_intensity = max(0.0, self.blur_intensity - 0.25)
                    self.last_logged_violation = None

                # Drawing landmarks - since we use custom Landmark class, we need a small adaptation
                # but mp drawing_utils usually expects proto objects.
                # For simplicity in preview, we'll draw manually if needed,
                # but let's try if it accepts our objects.
                if self.show_preview:
                    try:
                        drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                    except:
                        # Fallback: draw basic points if mp.drawing fails with custom objects
                        h, w, _ = frame.shape
                        for lm in landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
            else:
                self.blur_intensity = max(0.0, self.blur_intensity - 0.1)
                self.last_logged_violation = None
            if self.overlay: self.overlay.set_alpha(self.blur_intensity)
            self.update_tray_status(status_color)
            if self.show_preview:
                if self.blur_intensity > 0:
                    blur_amt = int(51 * self.blur_intensity)
                    if blur_amt % 2 == 0: blur_amt += 1
                    frame = cv2.GaussianBlur(frame, (blur_amt, blur_amt), 0)
                if self.calibrate_requested: cv2.putText(frame, "CALIBRATING...", (50, 50), 1, 2, (0,255,255), 2)
                if violation: cv2.putText(frame, f"VIOLATION: {violation[0]}", (50, 100), 1, 1.5, (0,0,255), 2)
                cv2.putText(
                    frame,
                    f"Provider: {self.detector.provider_short().upper()} | FPS: {current_fps:.1f}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                cv2.imshow("Posture Sentinel", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): self.toggle_preview(None, None)
                elif key == ord('c'): self.calibrate_requested = True
            else: cv2.destroyAllWindows(); cv2.waitKey(1)
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Posture Sentinel")
    parser.add_argument("--headless", action="store_true", help="Run without preview window")
    parser.add_argument("--no-tray", action="store_true", help="Run without system tray icon")
    args = parser.parse_args()

    app = PostureApp(headless=args.headless, no_tray=args.no_tray)
    cv_thread = threading.Thread(target=app.run_cv, daemon=True)
    overlay_thread = threading.Thread(target=app.run_overlay, daemon=True)
    cv_thread.start()
    overlay_thread.start()

    if args.no_tray:
        try:
            while app.running:
                time.sleep(0.5)
        except KeyboardInterrupt:
            app.running = False
            if app.overlay:
                app.overlay.stop()
            app.export_daily_summary()
    else:
        app.run_tray()
