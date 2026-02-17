import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional
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

@dataclass
class Config:
    camera_id: int = 0
    slouch_threshold: float = 0.05
    lean_threshold: float = 0.15
    violation_timeout_sec: float = 3.0
    show_preview: bool = True
    overlay_color: str = "white"

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
    def __init__(self, color="white"):
        self.root = tk.Tk()
        self.root.title("PostureSentinelOverlay")
        self.root.attributes("-topmost", True)
        self.root.attributes("-alpha", 0.0)
        self.root.attributes("-fullscreen", True)
        self.root.config(bg=color)
        self.root.overrideredirect(True)

        # Make window transparent for clicks
        hwnd = win32gui.FindWindow(None, "PostureSentinelOverlay")
        ex_style = win32gui.GetWindowLong(hwnd, win32con.GWL_EXSTYLE)
        win32gui.SetWindowLong(hwnd, win32con.GWL_EXSTYLE,
                               ex_style | win32con.WS_EX_LAYERED | win32con.WS_EX_TRANSPARENT)

        self.target_alpha = 0.0
        self.current_alpha = 0.0

    def set_alpha(self, alpha):
        self.target_alpha = max(0.0, min(0.6, alpha)) # Cap at 60% opacity

    def update(self):
        if abs(self.current_alpha - self.target_alpha) > 0.005:
            step = 0.01 if self.target_alpha > self.current_alpha else 0.03
            if self.current_alpha < self.target_alpha:
                self.current_alpha = min(self.target_alpha, self.current_alpha + step)
            else:
                self.current_alpha = max(self.target_alpha, self.current_alpha - step)
            self.root.attributes("-alpha", self.current_alpha)

        self.root.after(30, self.update)

    def run(self):
        self.update()
        self.root.mainloop()

    def stop(self):
        self.root.destroy()

class PostureAnalyzer:
    def __init__(self):
        self.baseline: Optional[dict] = None
        self.violation_start: Optional[float] = None
        self.current_violation: Optional[str] = None

    def calibrate(self, landmarks):
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_ear = landmarks[7]
        right_ear = landmarks[8]

        self.baseline = {
            "shoulder_y": (left_shoulder.y + right_shoulder.y) / 2,
            "ear_distance": abs(left_ear.x - right_ear.x),
            "shoulder_center_x": (left_shoulder.x + right_shoulder.x) / 2,
            "ear_center_x": (left_ear.x + right_ear.x) / 2,
        }

    def check(self, landmarks, timeout_sec: float = 3.0) -> Optional[tuple[str, float]]:
        if not self.baseline:
            return None

        current_time = time.time()
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_ear = landmarks[7]
        right_ear = landmarks[8]

        current_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        current_ear_distance = abs(left_ear.x - right_ear.x)

        shoulder_diff = current_shoulder_y - self.baseline["shoulder_y"]
        ear_distance_ratio = current_ear_distance / self.baseline["ear_distance"]

        violation = None
        if shoulder_diff > 0.05:
            violation = "slouching"
        elif ear_distance_ratio > 1.15:
            violation = "leaning_forward"

        if violation:
            if self.current_violation == violation and self.violation_start:
                elapsed = current_time - self.violation_start
                if elapsed >= timeout_sec:
                    return (violation, elapsed)
            else:
                self.current_violation = violation
                self.violation_start = current_time
        else:
            self.current_violation = None
            self.violation_start = None

        return None

def apply_blur_effect(frame, intensity: float = 0.5):
    h, w = frame.shape[:2]
    blur_amount = int(51 * intensity)
    if blur_amount % 2 == 0:
        blur_amount += 1
    return cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)

class PostureApp:
    def __init__(self):
        self.config = Config.load()
        self.analyzer = PostureAnalyzer()
        self.running = True
        self.icon: Optional[pystray.Icon] = None
        self.show_preview = self.config.show_preview
        self.calibrate_requested = False
        self.overlay = None
        self.blur_intensity = 0.0

    def create_icon_image(self, color="green"):
        image = Image.new('RGB', (64, 64), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        draw.ellipse((8, 8, 56, 56), fill=color)
        return image

    def on_exit(self, icon, item):
        self.running = False
        if self.overlay:
            self.overlay.root.after(0, self.overlay.stop)
        icon.stop()

    def toggle_preview(self, icon, item):
        self.show_preview = not self.show_preview
        self.config.show_preview = self.show_preview
        self.config.save()

    def request_calibration(self, icon, item):
        self.calibrate_requested = True

    def run_tray(self):
        menu = pystray.Menu(
            pystray.MenuItem("Show/Hide Preview", self.toggle_preview, checked=lambda item: self.show_preview),
            pystray.MenuItem("Calibrate", self.request_calibration),
            pystray.MenuItem("Exit", self.on_exit)
        )
        self.icon = pystray.Icon("PostureSentinel", self.create_icon_image(), "Posture Sentinel", menu)
        self.icon.run()

    def run_overlay(self):
        self.overlay = Overlay(self.config.overlay_color)
        self.overlay.run()

    def run_cv(self):
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose(model_complexity=1)
        drawing = mp.solutions.drawing_utils
        cap = cv2.VideoCapture(self.config.camera_id)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            frame = cv2.flip(frame, 1)
            results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            status_color = "green"
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                if self.calibrate_requested:
                    self.analyzer.calibrate(landmarks)
                    self.config.save()
                    self.calibrate_requested = False

                violation = self.analyzer.check(landmarks, self.config.violation_timeout_sec)
                if violation:
                    status_color = "red"
                    self.blur_intensity = min(1.0, self.blur_intensity + 0.05)
                elif self.analyzer.current_violation:
                    status_color = "orange"
                else:
                    self.blur_intensity = max(0.0, self.blur_intensity - 0.05)

                if self.show_preview:
                    drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            else:
                self.blur_intensity = max(0.0, self.blur_intensity - 0.05)

            if self.overlay:
                self.overlay.set_alpha(self.blur_intensity)
            if self.icon:
                self.icon.icon = self.create_icon_image(status_color)

            if self.show_preview:
                if self.blur_intensity > 0:
                    frame = apply_blur_effect(frame, self.blur_intensity)
                cv2.imshow("Posture Sentinel", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.toggle_preview(None, None)
            else:
                cv2.destroyAllWindows()
                cv2.waitKey(1)

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = PostureApp()
    threading.Thread(target=app.run_cv, daemon=True).start()
    threading.Thread(target=app.run_overlay, daemon=True).start()
    app.run_tray()
