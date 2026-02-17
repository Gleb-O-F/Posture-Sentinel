import cv2
import mediapipe as mp
import numpy as np
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
    def __init__(self):
        self.config = Config.load()
        self.analyzer = PostureAnalyzer(self.config.baseline)
        self.running, self.show_preview, self.calibrate_requested = True, self.config.show_preview, False
        self.icon, self.overlay, self.blur_intensity = None, None, 0.0

    def create_icon_image(self, color="green"):
        img = Image.new('RGBA', (64, 64), (0, 0, 0, 0))
        ImageDraw.Draw(img).ellipse((4, 4, 60, 60), fill=color)
        return img

    def on_exit(self, icon, item):
        self.running = False
        if self.overlay: self.overlay.stop()
        icon.stop()
        os._exit(0)

    def toggle_preview(self, icon, item):
        self.show_preview = not self.show_preview
        self.config.show_preview = self.show_preview
        self.config.save()

    def run_overlay(self):
        self.overlay = Overlay(self.config.overlay_color)
        self.overlay.run()

    def run_tray(self):
        menu = pystray.Menu(pystray.MenuItem("Show/Hide Preview", self.toggle_preview, checked=lambda item: self.show_preview),
                            pystray.MenuItem("Calibrate", lambda: setattr(self, 'calibrate_requested', True)),
                            pystray.MenuItem("Exit", self.on_exit))
        self.icon = pystray.Icon("PostureSentinel", self.create_icon_image(), "Posture Sentinel", menu)
        self.icon.run()

    def run_cv(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(model_complexity=1)
        drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(self.config.camera_id)
        while self.running:
            ret, frame = cap.read()
            if not ret: time.sleep(0.1); continue
            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
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
                elif self.analyzer.current_violation:
                    status_color = "orange"
                else:
                    self.blur_intensity = max(0.0, self.blur_intensity - 0.25)
                if self.show_preview: drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else: self.blur_intensity = max(0.0, self.blur_intensity - 0.1)
            if self.overlay: self.overlay.set_alpha(self.blur_intensity)
            if self.icon: self.icon.icon = self.create_icon_image(status_color)
            if self.show_preview:
                if self.blur_intensity > 0:
                    blur_amt = int(51 * self.blur_intensity)
                    if blur_amt % 2 == 0: blur_amt += 1
                    frame = cv2.GaussianBlur(frame, (blur_amt, blur_amt), 0)
                if self.calibrate_requested: cv2.putText(frame, "CALIBRATING...", (50, 50), 1, 2, (0,255,255), 2)
                if violation: cv2.putText(frame, f"VIOLATION: {violation[0]}", (50, 100), 1, 1.5, (0,0,255), 2)
                cv2.imshow("Posture Sentinel", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): self.toggle_preview(None, None)
                elif key == ord('c'): self.calibrate_requested = True
            else: cv2.destroyAllWindows(); cv2.waitKey(1)
        cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PostureApp()
    threading.Thread(target=app.run_cv, daemon=True).start()
    threading.Thread(target=app.run_overlay, daemon=True).start()
    app.run_tray()
