import cv2
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Optional
import yaml
from pathlib import Path

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
                data = yaml.safe_load(f)
                return cls(**data)
        return cls()
    
    def save(self, path: str = "config.yaml"):
        with open(path, "w") as f:
            yaml.dump(self.__dict__, f)

class PostureAnalyzer:
    def __init__(self):
        self.baseline: Optional[dict] = None
        self.violation_start: Optional[float] = None
        
    def calibrate(self, landmarks):
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        
        self.baseline = {
            "shoulder_y": (left_shoulder.y + right_shoulder.y) / 2,
            "nose_y": landmarks[0].y,
        }
        
    def check(self, landmarks) -> Optional[str]:
        if not self.baseline:
            return None
            
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        current_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2
        
        diff = current_shoulder_y - self.baseline["shoulder_y"]
        
        if diff > self.baseline["shoulder_y"] * 0.05:
            return "slouching"
            
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

def main():
    config = Config.load()
    
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
    )
    
    cap = cv2.VideoCapture(config.camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    analyzer = PostureAnalyzer()
    
    connections = list(mp_pose.POSE_CONNECTIONS)
    
    print("Posture Sentinel started")
    print("Press 'c' to calibrate, 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            draw_skeleton(frame, landmarks, connections)
            
            violation = analyzer.check(landmarks)
            
            if violation == "slouching":
                cv2.putText(frame, "BAD POSTURE!", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            elif analyzer.baseline:
                cv2.putText(frame, "Good", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        
        cv2.imshow("Posture Sentinel", frame)
        
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
