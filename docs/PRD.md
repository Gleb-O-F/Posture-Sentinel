# PRD: Posture Sentinel (Python Edition)

## 1. Product Overview

Posture Sentinel is a lightweight Windows desktop application that uses a webcam and local computer vision inference to monitor user posture in real time. When posture violations persist, the app gradually increases a screen blur overlay as soft feedback.

## 2. Current Goals

| Goal | Description |
|------|-------------|
| Performance | Keep realtime inference usable on Windows laptops and workstations with DirectML preferred and CPU fallback available |
| Reliability | Survive camera loss, provider downgrade, and missing history without hard crashes |
| Privacy | Process video locally with no cloud transfer |
| Operability | Support tray mode, headless mode, daily summaries, and threshold tuning from local logs |

## 3. Current Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| Video | OpenCV |
| Pose inference | ONNX Runtime (`onnxruntime-directml` on Windows, CPU fallback) |
| Landmark utilities | MediaPipe drawing helpers |
| Desktop UX | Tkinter overlay, `pystray`, Win32 APIs |
| Reporting | Local JSONL logs, performance logs, and JSON summaries |

## 4. Functional Scope

### 4.1 Tracking

The app extracts posture landmarks and focuses on:
- shoulders
- ears
- nose

### 4.2 Calibration

The user can save a baseline posture. Baseline metrics are persisted in `config.yaml`.

### 4.3 Posture Violations

The app currently detects:
- slouching
- leaning forward
- shoulder tilt
- body rotation

A violation is confirmed only after it lasts longer than the configured timeout.

### 4.4 Feedback

The app provides:
- fullscreen blur overlay with click-through behavior
- system tray status indicator
- optional preview window with provider and FPS

### 4.5 Operations and Analytics

The app supports:
- `--headless`
- `--no-tray`
- JSONL violation logs in `logs/`
- daily summary export
- performance telemetry and perf summary CLI
- threshold auto-tuning from recent history

## 5. Next Delivery Targets

1. Add automated smoke coverage for config loading, reporting, and tuning modules.
2. Harden startup and runtime error handling around camera, overlay, and tray initialization.
3. Measure DirectML vs CPU behavior on target hardware and tune fallback thresholds from real runs.
4. Decide whether Rust remains a future migration target or should be removed from planning artifacts.
