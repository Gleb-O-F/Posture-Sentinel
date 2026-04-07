import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from dataclasses import dataclass, field
from typing import Optional, Dict
from pathlib import Path
import time
import json
import threading
import pystray
from PIL import Image, ImageDraw
import win32gui
import win32con
import tkinter as tk
import os
import sys
import ctypes
import argparse
from datetime import datetime
from reporting import build_daily_summary, write_summary_file
from posture_quality_reporting import build_daily_quality_summary, write_quality_summary_file
from tuning import tune_thresholds

class AppInitError(RuntimeError):
    pass

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
    slouch_threshold: float = 0.15     # Neck compression threshold relative to shoulder width
    lean_threshold: float = 0.15       # Forward lean threshold based on ear distance
    shoulder_tilt_threshold: float = 0.10 # Shoulder tilt threshold
    shoulder_width_threshold: float = 0.8  # Body rotation threshold
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
    posture_sensitivity: float = 1.0
    metric_smoothing_alpha: float = 0.35
    bad_posture_enter_score: float = 1.0
    bad_posture_exit_score: float = 0.72
    good_posture_score_threshold: float = 0.45
    straight_posture_threshold: float = 0.28
    min_tracking_confidence: float = 0.55
    min_calibration_tracking_confidence: float = 0.7
    quality_advice_enabled: bool = True
    away_timeout_sec: float = 2.5
    break_reminder_interval_sec: float = 2700.0

    @staticmethod
    def normalize_baseline(baseline: Optional[Dict[str, float]]) -> Optional[Dict[str, float]]:
        if not isinstance(baseline, dict):
            return None

        if {"neck_dist", "ear_ratio", "tilt", "width"}.issubset(baseline.keys()):
            normalized = dict(baseline)
            try:
                normalized["head_height"] = float(normalized.get("head_height", normalized["neck_dist"]))
                normalized["nose_offset"] = float(normalized.get("nose_offset", 0.0))
            except (TypeError, ValueError):
                return None
            return normalized

        legacy_map = {
            "shoulder_y_norm": "neck_dist",
            "ear_dist_ratio": "ear_ratio",
            "shoulder_tilt_base": "tilt",
            "shoulder_width_base": "width",
        }
        if not set(legacy_map.keys()).issubset(baseline.keys()):
            return baseline

        return None

    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        config_path = Path(path)
        if config_path.exists():
            try:
                with open(config_path, encoding="utf-8") as f:
                    data = json.load(f)
                valid_keys = cls.__dataclass_fields__.keys()
                filtered_data = {k: v for k, v in data.items() if k in valid_keys}
                raw_baseline = filtered_data.get("baseline")
                filtered_data["baseline"] = cls.normalize_baseline(filtered_data.get("baseline"))
                if raw_baseline and filtered_data["baseline"] is None:
                    print("Legacy baseline detected in config.yaml; recalibration is required.")
                    config = cls(**filtered_data)
                    config.save(path)
                    return config
                return cls(**filtered_data)
            except Exception as e:
                print(f"Warning: failed to load config from {config_path}: {e}")
                backup_path = cls.backup_problematic_config(config_path)
                default_config = cls()
                default_config.save(path)
                if backup_path is not None:
                    print(f"Problematic config was backed up to {backup_path}")
        return cls()

    @staticmethod
    def backup_problematic_config(config_path: Path) -> Optional[Path]:
        if not config_path.exists():
            return None
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        backup_path = config_path.with_suffix(f"{config_path.suffix}.broken-{timestamp}.bak")
        try:
            with open(config_path, "r", encoding="utf-8", errors="replace") as src, open(
                backup_path, "w", encoding="utf-8"
            ) as dst:
                dst.write(src.read())
            return backup_path
        except Exception as backup_error:
            print(f"Warning: failed to back up problematic config: {backup_error}")
            return None

    def save(self, path: str = "config.yaml"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.__dict__, f, ensure_ascii=False, indent=2)

class Overlay:
    def __init__(self, color="#FFFFFF"):
        self.root = None
        self.color = color
        self.target_alpha = 0.0
        self.current_alpha = 0.0
        self.running = True
        self.warning_visible = False
        self.warning_label = None

    def set_alpha(self, alpha):
        self.target_alpha = max(0.0, min(0.85, alpha))

    def set_warning_visible(self, visible: bool):
        self.warning_visible = visible
        if self.root:
            try:
                self.root.after(0, self._apply_warning_state)
            except tk.TclError:
                pass

    def _apply_warning_state(self):
        if not self.warning_label:
            return
        if self.warning_visible:
            self.warning_label.lift()
            self.warning_label.place(relx=0.5, rely=0.5, anchor="center")
        else:
            self.warning_label.place_forget()

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
        self._apply_warning_state()
        self.root.after(20, self.update)

    def run(self):
        self.root = tk.Tk()
        self.root.title("PostureSentinelOverlay")
        self.root.attributes("-topmost", True, "-alpha", 0.0, "-fullscreen", True)
        self.root.config(bg=self.color)
        self.root.overrideredirect(True)
        self.warning_label = tk.Label(
            self.root,
            text="НЕРОВНАЯ ОСАНКА",
            font=("Arial", 52, "bold"),
            fg="#C1121F",
            bg=self.color,
            padx=48,
            pady=24,
        )
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


@dataclass
class PostureEvaluation:
    posture_state: str
    posture_score: float = 0.0
    tracking_score: float = 0.0
    strongest_violation: Optional[str] = None
    strongest_score: float = 0.0
    deviation_scores: Dict[str, float] = field(default_factory=dict)
    confirmed_violation: Optional[tuple[str, float]] = None
    calibration_allowed: bool = False
    calibration_reason: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None

class ONNXPoseDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.session = None
        self.active_provider = "uninitialized"
        if not Path(model_path).exists():
            raise AppInitError(f"ONNX model not found: {model_path}")
        self._init_session(prefer_dml=True)

    def _init_session(self, prefer_dml: bool):
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider'] if prefer_dml else ['CPUExecutionProvider']
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            active_providers = self.session.get_providers()
            self.active_provider = active_providers[0] if active_providers else "unknown"
            print(f"ONNX Session initialized. Active providers: {active_providers}")
        except Exception as e:
            if not prefer_dml:
                raise AppInitError(f"Failed to initialize ONNX Runtime session: {e}") from e
            print(f"Warning: Failed to initialize ONNX with DML, falling back to CPU: {e}")
            try:
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                self.active_provider = "CPUExecutionProvider"
            except Exception as cpu_error:
                raise AppInitError(f"Failed to initialize ONNX Runtime CPU session: {cpu_error}") from cpu_error

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
        self.smoothed_metrics: Optional[Dict[str, float]] = None
        self.current_posture_score: float = 0.0
        self.current_tracking_score: float = 0.0
        self.current_posture_state: str = "uncalibrated"
        self.calibration_message: Optional[str] = None

    def compute_metrics(self, landmarks) -> Optional[Dict[str, float]]:
        ls, rs, le, re, nose = landmarks[11], landmarks[12], landmarks[7], landmarks[8], landmarks[0]
        width = abs(ls.x - rs.x)
        if width <= 0.05:
            return None

        shoulder_mid_x = (ls.x + rs.x) / 2
        shoulder_mid_y = (ls.y + rs.y) / 2
        head_height = (shoulder_mid_y - nose.y) / width

        return {
            "head_height": head_height,
            "neck_dist": head_height,
            "ear_ratio": abs(le.x - re.x) / width,
            "tilt": (ls.y - rs.y) / width,
            "width": width,
            "nose_offset": abs(nose.x - shoulder_mid_x) / width,
        }

    def compute_tracking_score(self, landmarks) -> float:
        key_indices = (0, 7, 8, 11, 12)
        scores = []
        for index in key_indices:
            point = landmarks[index]
            visibility = 1.0 if point.visibility is None else float(point.visibility)
            presence = visibility if point.presence is None else float(point.presence)
            scores.append(max(0.0, min(1.0, min(visibility, presence))))
        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def smooth_metrics(self, metrics: Dict[str, float], alpha: float) -> Dict[str, float]:
        alpha = max(0.05, min(1.0, alpha))
        if self.smoothed_metrics is None:
            self.smoothed_metrics = dict(metrics)
            return dict(metrics)

        smoothed = {}
        for key, value in metrics.items():
            previous = self.smoothed_metrics.get(key, value)
            smoothed[key] = previous + alpha * (value - previous)
        self.smoothed_metrics = smoothed
        return smoothed

    def reset_violation_state(self):
        self.current_violation = None
        self.violation_start = None

    def calibrate(self, landmarks, config: Config) -> Optional[Dict[str, float]]:
        metrics = self.compute_metrics(landmarks)
        tracking_score = self.compute_tracking_score(landmarks)
        if metrics is None:
            self.calibration_message = "Move closer to the camera and keep shoulders visible."
            return None
        if tracking_score < config.min_calibration_tracking_confidence:
            self.calibration_message = "Tracking quality is too low for calibration."
            return None
        self.baseline = metrics
        self.smoothed_metrics = dict(metrics)
        self.current_posture_score = 0.0
        self.current_tracking_score = tracking_score
        self.current_posture_state = "good"
        self.calibration_message = "Calibration captured successfully."
        self.reset_violation_state()
        return self.baseline

    def evaluate(self, landmarks, config: Config) -> PostureEvaluation:
        tracking_score = self.compute_tracking_score(landmarks)
        self.current_tracking_score = tracking_score

        if not self.baseline:
            self.current_posture_state = "uncalibrated"
            self.current_posture_score = 0.0
            self.reset_violation_state()
            return PostureEvaluation(
                posture_state="uncalibrated",
                tracking_score=tracking_score,
                calibration_allowed=tracking_score >= config.min_calibration_tracking_confidence,
                calibration_reason="Press C while sitting straight to calibrate.",
            )

        metrics = self.compute_metrics(landmarks)
        if metrics is None:
            self.current_posture_state = "tracking_low"
            self.current_posture_score = 0.0
            self.reset_violation_state()
            return PostureEvaluation(
                posture_state="tracking_low",
                tracking_score=tracking_score,
                calibration_allowed=False,
                calibration_reason="Keep your shoulders and head fully in frame.",
            )

        if tracking_score < config.min_tracking_confidence:
            self.current_posture_state = "tracking_low"
            self.current_posture_score = 0.0
            self.reset_violation_state()
            return PostureEvaluation(
                posture_state="tracking_low",
                tracking_score=tracking_score,
                calibration_allowed=False,
                calibration_reason="Tracking quality is low. Face the camera and sit still.",
                metrics=metrics,
            )

        metrics = self.smooth_metrics(metrics, config.metric_smoothing_alpha)
        if not self.baseline:
            return PostureEvaluation(posture_state="uncalibrated", tracking_score=tracking_score)

        baseline_head_height = float(self.baseline.get("head_height", self.baseline.get("neck_dist", 0.0)))
        baseline_ear_ratio = float(self.baseline.get("ear_ratio", 0.0))
        baseline_tilt = float(self.baseline.get("tilt", 0.0))
        baseline_width = float(self.baseline.get("width", 0.0))
        baseline_nose_offset = float(self.baseline.get("nose_offset", 0.0))
        if baseline_width <= 0:
            self.current_posture_state = "uncalibrated"
            return PostureEvaluation(posture_state="uncalibrated", tracking_score=tracking_score, metrics=metrics)

        sensitivity = max(0.65, min(1.5, config.posture_sensitivity))
        slouch_threshold = max(config.slouch_threshold / sensitivity, 0.08)
        lean_threshold = max(config.lean_threshold / sensitivity, 0.12)
        tilt_threshold = max(config.shoulder_tilt_threshold / sensitivity, 0.08)
        rotation_threshold = max(config.shoulder_width_threshold - ((sensitivity - 1.0) * 0.08), 0.72)

        deviation_scores = {
            "slouching": max(0.0, (baseline_head_height - metrics["head_height"]) / slouch_threshold),
            "leaning_forward": 0.0,
            "shoulder_tilt": abs(metrics["tilt"] - baseline_tilt) / tilt_threshold,
            "body_rotation": max(0.0, (rotation_threshold - (metrics["width"] / baseline_width)) / 0.04),
        }

        ear_ratio_delta = metrics["ear_ratio"] - baseline_ear_ratio
        nose_offset_delta = metrics["nose_offset"] - baseline_nose_offset
        if ear_ratio_delta > 0:
            deviation_scores["leaning_forward"] = max(
                0.0,
                (ear_ratio_delta + max(0.0, nose_offset_delta) * 0.5) / lean_threshold,
            )

        strongest_violation = max(deviation_scores, key=deviation_scores.get)
        strongest_score = deviation_scores[strongest_violation]
        aggregate_score = sum(min(score, 1.5) for score in deviation_scores.values())
        posture_score = max(strongest_score, aggregate_score / 2.0)
        enter_score = max(config.bad_posture_enter_score, config.good_posture_score_threshold + 0.15)
        exit_score = min(enter_score - 0.1, max(config.bad_posture_exit_score, config.good_posture_score_threshold + 0.1))
        straight_score_threshold = max(
            0.1,
            min(config.straight_posture_threshold, config.good_posture_score_threshold, exit_score - 0.05),
        )
        confirmed_violation = None
        started_new_violation = False
        active_elapsed: Optional[float] = None

        if self.current_violation:
            active_score = deviation_scores.get(self.current_violation, strongest_score)
            if active_score >= exit_score or posture_score >= exit_score:
                if self.violation_start:
                    active_elapsed = time.time() - self.violation_start
                    if active_elapsed >= config.violation_timeout_sec:
                        confirmed_violation = (self.current_violation, active_elapsed)
            else:
                self.reset_violation_state()

        if self.current_violation is None and posture_score >= enter_score and strongest_score >= enter_score * 0.9:
            self.current_violation = strongest_violation
            self.violation_start = time.time()
            started_new_violation = True

        posture_state = "good"
        if self.current_violation:
            if started_new_violation:
                posture_state = "pending"
            elif self.violation_start:
                elapsed = active_elapsed if active_elapsed is not None else time.time() - self.violation_start
                if elapsed >= config.violation_timeout_sec:
                    confirmed_violation = (self.current_violation, elapsed)
                    posture_state = "bad"
                else:
                    posture_state = "pending"
            else:
                posture_state = "pending"
        elif posture_score > straight_score_threshold:
            posture_state = "pending"
        else:
            posture_state = "good"

        self.current_posture_score = posture_score
        self.current_posture_state = posture_state
        return PostureEvaluation(
            posture_state=posture_state,
            posture_score=posture_score,
            tracking_score=tracking_score,
            strongest_violation=strongest_violation,
            strongest_score=strongest_score,
            deviation_scores=deviation_scores,
            confirmed_violation=confirmed_violation,
            calibration_allowed=tracking_score >= config.min_calibration_tracking_confidence,
            calibration_reason="Press C while sitting straight to recalibrate.",
            metrics=metrics,
        )

    def check(self, landmarks, config: Config) -> Optional[tuple[str, float]]:
        evaluation = self.evaluate(landmarks, config)
        return evaluation.confirmed_violation

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
        self.overlay_available = True
        self.tray_available = not no_tray
        self.last_perf_log_provider: Optional[str] = None
        self.status_message: Optional[str] = None
        self.quality_advice: Optional[str] = None
        self.last_logged_quality_advice: Optional[str] = None
        self.low_tracking_streak = 0
        self.pending_posture_streak = 0
        self.break_reminder_message: Optional[str] = None
        self.break_message_expires_at = 0.0
        self.no_pose_since: Optional[float] = None
        self.is_user_away = False
        self.session_started_at = time.time()
        self.session_last_state_at = self.session_started_at
        self.session_state_durations: Dict[str, float] = {
            "good": 0.0,
            "pending": 0.0,
            "bad": 0.0,
            "tracking_low": 0.0,
            "uncalibrated": 0.0,
            "away": 0.0,
        }
        self.session_present_seconds = 0.0
        self.session_away_count = 0
        self.session_return_count = 0
        self.session_break_reminders = 0
        self.last_break_reset_at = self.session_started_at
        self.last_present_at = self.session_started_at

        model_path = Path("models") / "pose_landmarks_detector_full.onnx"
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

    def log_violation(self, violation_type: str, duration_sec: float, evaluation: Optional[PostureEvaluation] = None):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now()
        log_file = self.logs_dir / f"{ts.strftime('%Y-%m-%d')}.jsonl"
        entry = {
            "timestamp": ts.isoformat(timespec="seconds"),
            "violation": violation_type,
            "duration_sec": round(duration_sec, 2),
            "camera_status": self.camera_status,
            "posture_state": evaluation.posture_state if evaluation else self.analyzer.current_posture_state,
            "posture_score": round(evaluation.posture_score if evaluation else self.analyzer.current_posture_score, 3),
            "tracking_score": round(evaluation.tracking_score if evaluation else self.analyzer.current_tracking_score, 3),
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

    def export_quality_summary(self, day: Optional[str] = None) -> Optional[Path]:
        target_day = day or datetime.now().strftime("%Y-%m-%d")
        summary = build_daily_quality_summary(self.logs_dir, target_day)
        if summary is None:
            print(f"Quality summary skipped: no log file for {target_day}")
            return None

        summary_file = write_quality_summary_file(self.logs_dir, f"{target_day}.quality.summary.json", summary)
        print(f"Quality summary exported: {summary_file}")
        return summary_file


    def log_runtime_event(self, event_type: str, **payload):
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now()
        log_file = self.logs_dir / f"{ts.strftime('%Y-%m-%d')}.perf.jsonl"
        entry = {
            "timestamp": ts.isoformat(timespec="seconds"),
            "event": event_type,
            "provider": self.detector.provider_short() if self.detector else "n/a",
            "camera_status": self.camera_status,
        }
        entry.update(payload)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_performance_snapshot(self, fps: float):
        provider = self.detector.provider_short() if self.detector else "n/a"
        self.log_runtime_event(
            "performance",
            fps=round(fps, 2),
            preview_enabled=self.show_preview,
            headless=self.headless,
            overlay_available=self.overlay_available,
            tray_available=self.tray_available,
            provider_changed=provider != self.last_perf_log_provider,
            posture_state=self.analyzer.current_posture_state,
            posture_score=round(self.analyzer.current_posture_score, 3),
            tracking_score=round(self.analyzer.current_tracking_score, 3),
        )
        self.last_perf_log_provider = provider

    def log_provider_switch(self, previous_provider: str, new_provider: str, reason: str, fps: float):
        self.log_runtime_event(
            "provider_switch",
            previous_provider=previous_provider,
            new_provider=new_provider,
            reason=reason,
            fps=round(fps, 2),
        )

    def export_today_summary(self, icon=None, item=None):
        self.export_daily_summary()

    def export_today_quality_summary(self, icon=None, item=None):
        self.export_quality_summary()

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

    def adjust_posture_sensitivity(self, delta: float):
        self.config.posture_sensitivity = round(max(0.65, min(1.5, self.config.posture_sensitivity + delta)), 2)
        self.config.save()
        print(f"Posture sensitivity: {self.config.posture_sensitivity:.2f}")

    def increase_posture_sensitivity(self, icon=None, item=None):
        self.adjust_posture_sensitivity(0.1)

    def decrease_posture_sensitivity(self, icon=None, item=None):
        self.adjust_posture_sensitivity(-0.1)

    def adjust_good_posture_threshold(self, delta: float):
        self.config.good_posture_score_threshold = round(
            max(0.2, min(0.8, self.config.good_posture_score_threshold + delta)),
            2,
        )
        self.config.save()
        print(f"Good posture threshold: {self.config.good_posture_score_threshold:.2f}")

    def increase_good_posture_threshold(self, icon=None, item=None):
        self.adjust_good_posture_threshold(0.05)

    def decrease_good_posture_threshold(self, icon=None, item=None):
        self.adjust_good_posture_threshold(-0.05)

    def adjust_straight_posture_threshold(self, delta: float):
        self.config.straight_posture_threshold = round(
            max(0.1, min(0.6, self.config.straight_posture_threshold + delta)),
            2,
        )
        self.config.save()
        print(f"Straight posture threshold: {self.config.straight_posture_threshold:.2f}")

    def increase_straight_posture_threshold(self, icon=None, item=None):
        self.adjust_straight_posture_threshold(0.03)

    def decrease_straight_posture_threshold(self, icon=None, item=None):
        self.adjust_straight_posture_threshold(-0.03)

    def toggle_quality_advice(self, icon=None, item=None):
        self.config.quality_advice_enabled = not self.config.quality_advice_enabled
        self.config.save()
        if not self.config.quality_advice_enabled:
            self.quality_advice = None
        print(f"Quality advice enabled: {self.config.quality_advice_enabled}")

    def format_violation_reason(self, reason: Optional[str]) -> str:
        labels = {
            "slouching": "Slouching",
            "leaning_forward": "Leaning forward",
            "shoulder_tilt": "Shoulder tilt",
            "body_rotation": "Body rotation",
        }
        return labels.get(reason or "", "Posture drift")

    def update_presence_state(self, has_pose: bool, now: float) -> bool:
        if has_pose:
            if self.is_user_away:
                self.is_user_away = False
                self.session_return_count += 1
                self.last_break_reset_at = now
                self.log_runtime_event("user_returned")
            self.no_pose_since = None
            self.last_present_at = now
            return False

        if self.no_pose_since is None:
            self.no_pose_since = now
            return False

        if not self.is_user_away and (now - self.no_pose_since) >= self.config.away_timeout_sec:
            self.is_user_away = True
            self.session_away_count += 1
            self.last_logged_violation = None
            self.log_runtime_event("user_away")
        return self.is_user_away

    def update_session_metrics(self, state: str, now: float):
        elapsed = max(0.0, now - self.session_last_state_at)
        if state in self.session_state_durations:
            self.session_state_durations[state] += elapsed
        if state != "away":
            self.session_present_seconds += elapsed
        self.session_last_state_at = now

    def maybe_emit_break_reminder(self, now: float):
        if self.is_user_away or self.config.break_reminder_interval_sec <= 0:
            return
        if (now - self.last_break_reset_at) < self.config.break_reminder_interval_sec:
            return

        self.break_reminder_message = "Time for a short stretch break."
        self.break_message_expires_at = now + 10.0
        self.last_break_reset_at = now
        self.session_break_reminders += 1
        self.log_runtime_event("break_reminder", session_present_minutes=round(self.session_present_seconds / 60.0, 2))

    def get_session_summary(self) -> dict:
        total_session_sec = max(0.0, time.time() - self.session_started_at)
        return {
            "session_minutes": round(total_session_sec / 60.0, 2),
            "present_minutes": round(self.session_present_seconds / 60.0, 2),
            "away_count": self.session_away_count,
            "return_count": self.session_return_count,
            "break_reminders": self.session_break_reminders,
            "state_minutes": {
                key: round(value / 60.0, 2) for key, value in self.session_state_durations.items()
            },
        }

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

    def safe_log_runtime_event(self, event_type: str, **payload):
        logger = getattr(self, "log_runtime_event", None)
        if callable(logger):
            logger(event_type, **payload)

    def get_posture_feedback_state(self, evaluation: Optional[PostureEvaluation]) -> str:
        if evaluation is not None:
            return evaluation.posture_state
        if self.analyzer.current_violation:
            return "pending"
        if self.analyzer.baseline:
            return "good"
        return "uncalibrated"

    def update_quality_advice(self, evaluation: Optional[PostureEvaluation]):
        if not self.config.quality_advice_enabled or evaluation is None or getattr(self, "is_user_away", False):
            self.quality_advice = None
            self.low_tracking_streak = 0
            self.pending_posture_streak = 0
            return

        if evaluation.posture_state == "tracking_low":
            self.low_tracking_streak += 1
        else:
            self.low_tracking_streak = 0

        if evaluation.posture_state == "pending":
            self.pending_posture_streak += 1
        else:
            self.pending_posture_streak = 0

        if self.low_tracking_streak >= 20:
            self.quality_advice = "Tracking stays weak. Move closer, add light, and keep shoulders visible."
        elif self.pending_posture_streak >= 30:
            self.quality_advice = (
                "Posture is often borderline. Try recalibration or lower sensitivity if you are sitting straight."
            )
        elif (
            evaluation.posture_state == "good"
            and evaluation.tracking_score >= self.config.min_calibration_tracking_confidence
            and evaluation.posture_score <= self.config.good_posture_score_threshold * 0.7
        ):
            self.quality_advice = "Tracking looks stable. This is a good moment to recalibrate if needed."
        else:
            self.quality_advice = None

        last_logged_quality_advice = getattr(self, "last_logged_quality_advice", None)
        if self.quality_advice and self.quality_advice != last_logged_quality_advice:
            logger = getattr(self, "log_runtime_event", None)
            if callable(logger):
                logger("quality_advice", advice=self.quality_advice)
            self.last_logged_quality_advice = self.quality_advice
        elif self.quality_advice is None:
            self.last_logged_quality_advice = None

    def get_status_message(self, evaluation: Optional[PostureEvaluation]) -> Optional[str]:
        if self.break_reminder_message and time.time() <= self.break_message_expires_at:
            return self.break_reminder_message
        if self.is_user_away:
            return "Away from desk. Session timer is paused until you return."
        if self.quality_advice:
            return self.quality_advice
        if self.calibrate_requested:
            if self.analyzer.calibration_message:
                return self.analyzer.calibration_message
            return "Sit straight, keep shoulders visible, and hold still for calibration."
        if evaluation is None:
            return None
        if evaluation.posture_state == "tracking_low":
            return "Tracking is weak. Center yourself and keep your shoulders in frame."
        if evaluation.posture_state == "uncalibrated":
            return "Press C to calibrate while sitting straight."
        if evaluation.posture_state == "good":
            return (
                f"Straight posture detected. Score {evaluation.posture_score:.2f}, "
                f"tracking {evaluation.tracking_score:.2f}."
            )
        if evaluation.posture_state == "pending":
            return (
                f"Borderline posture: {self.format_violation_reason(evaluation.strongest_violation)}. Score {evaluation.posture_score:.2f}, "
                f"tracking {evaluation.tracking_score:.2f}."
            )
        if evaluation.confirmed_violation:
            return (
                f"Violation: {self.format_violation_reason(evaluation.confirmed_violation[0])}. Score {evaluation.posture_score:.2f}, "
                f"tracking {evaluation.tracking_score:.2f}."
            )
        return None

    def on_exit(self, icon, item):
        self.update_session_metrics("away" if self.is_user_away else self.analyzer.current_posture_state, time.time())
        self.log_runtime_event("session_summary", **self.get_session_summary())
        self.running = False
        if self.overlay:
            self.overlay.stop()
        self.export_daily_summary()
        self.export_quality_summary()
        if icon is not None:
            icon.stop()

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
        try:
            self.overlay = Overlay(self.config.overlay_color)
            self.overlay.run()
        except Exception as e:
            self.overlay = None
            self.overlay_available = False
            print(f"Warning: overlay disabled due to initialization failure: {e}")
            PostureApp.safe_log_runtime_event(self, "overlay_failure", error=str(e))

    def run_tray(self):
        try:
            menu = pystray.Menu(
                pystray.MenuItem("Show/Hide Preview", self.toggle_preview, checked=lambda item: self.show_preview),
                pystray.MenuItem("Auto Start", self.toggle_auto_start, checked=lambda item: self.config.auto_start),
                pystray.MenuItem("Auto Tune", self.toggle_auto_tune, checked=lambda item: self.config.auto_tune_enabled),
                pystray.MenuItem("Quality Advice", self.toggle_quality_advice, checked=lambda item: self.config.quality_advice_enabled),
                pystray.MenuItem("Sensitivity +", self.increase_posture_sensitivity),
                pystray.MenuItem("Sensitivity -", self.decrease_posture_sensitivity),
                pystray.MenuItem("Good Posture +", self.increase_good_posture_threshold),
                pystray.MenuItem("Good Posture -", self.decrease_good_posture_threshold),
                pystray.MenuItem("Straight Posture +", self.increase_straight_posture_threshold),
                pystray.MenuItem("Straight Posture -", self.decrease_straight_posture_threshold),
                pystray.MenuItem("Calibrate", self.request_calibration),
                pystray.MenuItem("Auto-tune Thresholds", self.auto_tune_thresholds),
                pystray.MenuItem("Export Daily Summary", self.export_today_summary),
                pystray.MenuItem("Export Quality Summary", self.export_today_quality_summary),
                pystray.MenuItem("Exit", self.on_exit),
            )
            self.icon = pystray.Icon("PostureSentinel", self.create_icon_image(), "Posture Sentinel", menu)
            self.icon.run()
            return True
        except Exception as e:
            self.icon = None
            self.tray_available = False
            print(f"Warning: tray disabled due to initialization failure: {e}")
            PostureApp.safe_log_runtime_event(self, "tray_failure", error=str(e))
            return False

    def run_cv(self):
        solutions = getattr(mp, "solutions", None)
        mp_pose = getattr(solutions, "pose", None) if solutions is not None else None
        drawing = getattr(solutions, "drawing_utils", None) if solutions is not None else None
        pose_connections = getattr(mp_pose, "POSE_CONNECTIONS", ()) if mp_pose is not None else ()
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
                self.log_performance_snapshot(current_fps)
                if (
                    self.config.enable_auto_cpu_fallback
                    and not cpu_fallback_done
                    and self.detector.active_provider == "DmlExecutionProvider"
                    and current_fps < self.config.min_fps_before_cpu_fallback
                ):
                    previous_provider = self.detector.provider_short()
                    if self.detector.fallback_to_cpu():
                        self.log_provider_switch(previous_provider, self.detector.provider_short(), "low_fps_auto_fallback", current_fps)
                        cpu_fallback_done = True
                        print(
                            f"Auto fallback: switched to CPU due to low FPS "
                            f"({current_fps:.1f} < {self.config.min_fps_before_cpu_fallback:.1f})"
                        )
                perf_window_start = now
                perf_frame_count = 0
            status_color = "green"
            evaluation: Optional[PostureEvaluation] = None
            violation = None
            current_runtime_state = "uncalibrated"
            has_pose = bool(results.pose_landmarks)
            user_away = self.update_presence_state(has_pose, now)
            if has_pose:
                landmarks = results.pose_landmarks.landmark
                if self.calibrate_requested:
                    baseline = self.analyzer.calibrate(landmarks, self.config)
                    if baseline is not None:
                        self.config.baseline = baseline
                        self.config.save()
                        self.calibrate_requested = False
                        print("Calibrated with updated posture baseline.")
                    else:
                        print(self.analyzer.calibration_message or "Calibration skipped.")
                evaluation = self.analyzer.evaluate(landmarks, self.config)
                violation = evaluation.confirmed_violation
                if user_away:
                    self.blur_intensity = max(0.0, self.blur_intensity - 0.2)
                    self.last_logged_violation = None
                    status_color = "gray"
                    current_runtime_state = "away"
                elif violation:
                    status_color = "red"
                    self.blur_intensity = min(1.0, self.blur_intensity + 0.08)
                    if self.last_logged_violation != violation[0]:
                        self.log_violation(violation[0], violation[1], evaluation)
                        self.last_logged_violation = violation[0]
                    current_runtime_state = "bad"
                elif evaluation.posture_state == "pending":
                    status_color = "orange"
                    self.blur_intensity = max(0.0, self.blur_intensity - 0.08)
                    self.last_logged_violation = None
                    current_runtime_state = "pending"
                elif evaluation.posture_state == "tracking_low":
                    status_color = "gray"
                    self.blur_intensity = max(0.0, self.blur_intensity - 0.15)
                    self.last_logged_violation = None
                    current_runtime_state = "tracking_low"
                else:
                    self.blur_intensity = max(0.0, self.blur_intensity - 0.25)
                    self.last_logged_violation = None
                    current_runtime_state = evaluation.posture_state

                # Drawing landmarks - since we use custom Landmark class, we need a small adaptation
                # but mp drawing_utils usually expects proto objects.
                # For simplicity in preview, we'll draw manually if needed,
                # but let's try if it accepts our objects.
                if self.show_preview:
                    try:
                        if drawing is None or not pose_connections:
                            raise AttributeError("mediapipe drawing utils unavailable")
                        drawing.draw_landmarks(frame, results.pose_landmarks, pose_connections)
                    except:
                        # Fallback: draw basic points if mp.drawing fails with custom objects
                        h, w, _ = frame.shape
                        for lm in landmarks:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)
            else:
                self.analyzer.current_tracking_score = 0.0
                self.analyzer.current_posture_state = "away" if user_away else ("tracking_low" if self.analyzer.baseline else "uncalibrated")
                self.blur_intensity = max(0.0, self.blur_intensity - 0.1)
                self.last_logged_violation = None
                current_runtime_state = self.analyzer.current_posture_state

            self.update_session_metrics(current_runtime_state, now)
            self.maybe_emit_break_reminder(now)

            posture_feedback_state = "away" if user_away else self.get_posture_feedback_state(evaluation)
            self.update_quality_advice(evaluation)
            self.status_message = self.get_status_message(evaluation)
            warning_visible = posture_feedback_state == "bad"
            if self.overlay:
                self.overlay.set_alpha(self.blur_intensity)
                self.overlay.set_warning_visible(warning_visible)
            self.update_tray_status(status_color)
            if self.show_preview:
                if self.blur_intensity > 0:
                    blur_amt = int(51 * self.blur_intensity)
                    if blur_amt % 2 == 0: blur_amt += 1
                    frame = cv2.GaussianBlur(frame, (blur_amt, blur_amt), 0)
                if self.calibrate_requested:
                    cv2.putText(frame, "CALIBRATING...", (50, 50), 1, 2, (0,255,255), 2)
                if violation:
                    cv2.putText(frame, f"VIOLATION: {violation[0]}", (50, 100), 1, 1.5, (0,0,255), 2)
                if warning_visible:
                    text = "НЕРОВНАЯ ОСАНКА"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 2.1
                    thickness = 5
                    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                    text_x = max((frame.shape[1] - text_size[0]) // 2, 20)
                    text_y = max(frame.shape[0] // 2, text_size[1] + 20)
                    cv2.putText(frame, text, (text_x + 4, text_y + 4), font, scale, (255, 255, 255), thickness + 2)
                    cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 0, 255), thickness)
                if posture_feedback_state == "good":
                    text = "ОСАНКА РОВНАЯ"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.8
                    thickness = 4
                    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                    text_x = max((frame.shape[1] - text_size[0]) // 2, 20)
                    text_y = max(frame.shape[0] // 2, text_size[1] + 20)
                    cv2.putText(frame, text, (text_x + 4, text_y + 4), font, scale, (0, 0, 0), thickness + 2)
                    cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 180, 0), thickness)
                elif posture_feedback_state == "tracking_low":
                    text = "TRACKING IS TOO WEAK"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.1
                    thickness = 3
                    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                    text_x = max((frame.shape[1] - text_size[0]) // 2, 20)
                    text_y = max(frame.shape[0] // 2, text_size[1] + 20)
                    cv2.putText(frame, text, (text_x + 3, text_y + 3), font, scale, (0, 0, 0), thickness + 2)
                    cv2.putText(frame, text, (text_x, text_y), font, scale, (120, 200, 255), thickness)
                elif posture_feedback_state == "uncalibrated":
                    text = "НУЖНА КАЛИБРОВКА (C)"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.3
                    thickness = 3
                    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                    text_x = max((frame.shape[1] - text_size[0]) // 2, 20)
                    text_y = max(frame.shape[0] // 2, text_size[1] + 20)
                    cv2.putText(frame, text, (text_x + 3, text_y + 3), font, scale, (0, 0, 0), thickness + 2)
                    cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 220, 255), thickness)
                elif posture_feedback_state == "away":
                    text = "AWAY FROM DESK"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.4
                    thickness = 3
                    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                    text_x = max((frame.shape[1] - text_size[0]) // 2, 20)
                    text_y = max(frame.shape[0] // 2, text_size[1] + 20)
                    cv2.putText(frame, text, (text_x + 3, text_y + 3), font, scale, (0, 0, 0), thickness + 2)
                    cv2.putText(frame, text, (text_x, text_y), font, scale, (180, 180, 180), thickness)
                elif posture_feedback_state == "pending":
                    text = "ADJUST YOUR POSTURE"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    scale = 1.4
                    thickness = 3
                    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
                    text_x = max((frame.shape[1] - text_size[0]) // 2, 20)
                    text_y = max(frame.shape[0] // 2, text_size[1] + 20)
                    cv2.putText(frame, text, (text_x + 3, text_y + 3), font, scale, (0, 0, 0), thickness + 2)
                    cv2.putText(frame, text, (text_x, text_y), font, scale, (0, 180, 255), thickness)
                if evaluation and evaluation.strongest_violation and posture_feedback_state in {"pending", "bad"}:
                    cv2.putText(
                        frame,
                        f"Reason: {self.format_violation_reason(evaluation.strongest_violation)}",
                        (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )
                if self.status_message:
                    cv2.putText(
                        frame,
                        self.status_message[:90],
                        (20, frame.shape[0] - 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 255),
                        2
                    )
                session_summary = self.get_session_summary()
                cv2.putText(
                    frame,
                    (
                        f"Session: {session_summary['session_minutes']:.1f}m | Present: {session_summary['present_minutes']:.1f}m | "
                        f"Breaks: {session_summary['break_reminders']} | Away: {session_summary['away_count']}"
                    ),
                    (20, frame.shape[0] - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    (
                        f"Provider: {self.detector.provider_short().upper()} | FPS: {current_fps:.1f} | "
                        f"Score: {self.analyzer.current_posture_score:.2f} | "
                        f"Track: {self.analyzer.current_tracking_score:.2f}"
                    ),
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
    parser.add_argument("--smoke-seconds", type=float, default=0.0, help="Stop automatically after the given number of seconds")
    args = parser.parse_args()

    try:
        app = PostureApp(headless=args.headless, no_tray=args.no_tray)
        cv_thread = threading.Thread(target=app.run_cv, daemon=True)
        overlay_thread = threading.Thread(target=app.run_overlay, daemon=True)
        cv_thread.start()
        overlay_thread.start()
        started_at = time.time()

        def should_stop_for_smoke() -> bool:
            return args.smoke_seconds > 0 and (time.time() - started_at) >= args.smoke_seconds

        def stop_app():
            app.update_session_metrics("away" if app.is_user_away else app.analyzer.current_posture_state, time.time())
            app.log_runtime_event("session_summary", **app.get_session_summary())
            app.running = False
            if app.overlay:
                app.overlay.stop()
            app.export_daily_summary()
            app.export_quality_summary()

        if args.no_tray:
            try:
                while app.running:
                    if should_stop_for_smoke():
                        stop_app()
                        break
                    time.sleep(0.5)
            except KeyboardInterrupt:
                stop_app()
        else:
            tray_started = app.run_tray()
            if not tray_started:
                try:
                    while app.running:
                        if should_stop_for_smoke():
                            stop_app()
                            break
                        time.sleep(0.5)
                except KeyboardInterrupt:
                    stop_app()
    except AppInitError as e:
        print(f"Startup failed: {e}")
        sys.exit(1)

