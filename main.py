import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import time
import math
import json

@dataclass
class Config:
    camera_id: int = 0
    slouch_threshold: float = 0.05
    lean_threshold: float = 0.15
    violation_timeout_sec: float = 3.0
    
    @classmethod
    def load(cls, path: str = "config.yaml") -> "Config":
        if Path(path).exists():
            with open(path) as f:
                data = json.load(f)
                return cls(**data)
        return cls()
    
    def save(self, path: str = "config.yaml"):
        with open(path, "w") as f:
            json.dump(self.__dict__, f)

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
        left_eye = landmarks[3]
        right_eye = landmarks[6]
        
        ear_distance = abs(left_ear.x - right_ear.x)
        
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        ear_center_x = (left_ear.x + right_ear.x) / 2
        
        self.baseline = {
            "shoulder_y": (left_shoulder.y + right_shoulder.y) / 2,
            "ear_y": (left_ear.y + right_ear.y) / 2,
            "eye_y": (left_eye.y + right_eye.y) / 2,
            "ear_distance": ear_distance,
            "shoulder_center_x": shoulder_center_x,
            "ear_center_x": ear_center_x,
        }
        
    def check(self, landmarks, timeout_sec: float = 3.0) -> Optional[tuple[str, float]]:
        if not self.baseline:
            return None
            
        current_time = time.time()
            
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_ear = landmarks[7]
        right_ear = landmarks[8]
        left_eye = landmarks[3]
        right_eye = landmarks[6]
        
        current_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        current_ear_y = (left_ear.y + right_ear.y) / 2
        current_eye_y = (left_eye.y + right_eye.y) / 2
        
        current_ear_distance = abs(left_ear.x - right_ear.x)
        
        shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
        ear_center_x = (left_ear.x + right_ear.x) / 2
        
        shoulder_diff = current_shoulder_y - self.baseline["shoulder_y"]
        ear_diff = current_ear_y - self.baseline["ear_y"]
        eye_diff = current_eye_y - self.baseline["eye_y"]
        
        ear_distance_ratio = current_ear_distance / self.baseline["ear_distance"]
        
        head_tilt_angle = abs(ear_center_x - shoulder_center_x) * 90
        
        violation = None
        
        if shoulder_diff > 0.05:
            violation = "slouching"
        elif ear_distance_ratio > 1.15:
            violation = "leaning_forward"
        elif head_tilt_angle > 15:
            violation = "head_tilt"
        elif ear_diff > 0.03:
            violation = "leaning_forward"
        elif eye_diff > 0.02:
            violation = "head_down"
        
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

def draw_skeleton(frame, landmarks, connections):
    h, w = frame.shape[:2]
    
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    
    for start, end in connections:
        if landmarks[start].visibility > 0.5 and landmarks[end].visibility > 0.5:
            x1 = int(landmarks[start].x * w)
            y1 = int(landmarks[start].y * h)
            x2 = int(landmarks[end].x * w)
            y2 = int(landmarks[end].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

def apply_blur_effect(frame, intensity: float = 0.5):
    h, w = frame.shape[:2]
    blur_amount = int(51 * intensity)
    if blur_amount % 2 == 0:
        blur_amount += 1
    return cv2.GaussianBlur(frame, (blur_amount, blur_amount), 0)

def main():
    config = Config.load()
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
    )
    drawing = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(config.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    analyzer = PostureAnalyzer()
    
    connections = list(mp_pose.POSE_CONNECTIONS)
    
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    blur_intensity = 0.0
    
    print("Posture Sentinel started")
    print("Press 'c' to calibrate, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        fps_counter += 1
        if time.time() - fps_start_time >= 1.0:
            fps = fps_counter
            fps_counter = 0
            fps_start_time = time.time()
        
        display_frame = frame.copy()
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            drawing.draw_landmarks(display_frame, results.pose_landmarks, connections)
            
            violation = analyzer.check(landmarks, config.violation_timeout_sec)
            
            if violation:
                violation_type, elapsed = violation
                cv2.putText(display_frame, f"BAD POSTURE: {violation_type.upper()}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(display_frame, f"Time: {elapsed:.1f}s / {config.violation_timeout_sec}s", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                blur_intensity = min(1.0, blur_intensity + 0.05)
            elif analyzer.current_violation:
                cv2.putText(display_frame, f"WARNING: {analyzer.current_violation.upper()}...", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                cv2.putText(display_frame, f"Wait {config.violation_timeout_sec}s to confirm", (50, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            elif analyzer.baseline:
                cv2.putText(display_frame, "Good posture", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                blur_intensity = max(0.0, blur_intensity - 0.1)
            else:
                cv2.putText(display_frame, "Press 'c' to calibrate", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
        else:
            blur_intensity = max(0.0, blur_intensity - 0.1)
        
        if blur_intensity > 0:
            display_frame = apply_blur_effect(display_frame, blur_intensity)
        
        cv2.putText(display_frame, f"FPS: {fps}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        cv2.imshow("Posture Sentinel", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and results.pose_landmarks:
            analyzer.calibrate(results.pose_landmarks.landmark)
            config.save()
            print("Calibrated!")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
