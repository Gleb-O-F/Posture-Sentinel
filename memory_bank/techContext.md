# Tech Context: Posture Sentinel

## Main Stack

- Python 3.10+
- OpenCV
- MediaPipe
- MediaPipe `tasks` package variant may be present without the legacy `solutions` API
- NumPy
- ONNX Runtime / ONNX Runtime DirectML
- Pillow
- pystray
- pywin32
- Tkinter

## Important Files

- `main.py`: runtime, inference, smoothed posture scoring, away detection, break reminders, session analytics, tray quality controls, UI orchestration, logging
- `reporting.py`: violation aggregation and summary serialization
- `performance_reporting.py`: performance telemetry aggregation
- `posture_quality_reporting.py`: posture score and tracking quality aggregation
- `tuning.py`: bounded threshold auto-tuning
- `tools/check_env.py`: environment diagnostics
- `tools/report.py`: violation summary CLI
- `tools/perf_report.py`: performance summary CLI
- `tools/quality_report.py`: posture quality summary CLI
- `tools/tune.py`: tuning CLI
- `tools/bootstrap.ps1`: environment bootstrap
- `tools/run.ps1`: guarded launch
- `tools/run_tests.ps1`: automated tests
- `tools/smoke_test.ps1`: smoke run

## Dependencies Source

Required Python packages are declared in `requirements.txt`.

## Current Environment Constraint

The repository is structurally ready and the current workspace now has a working Python interpreter for execution and smoke validation, but longer real-user sessions are still required for final performance review.

## Validation Target

A Windows workstation with Python 3.10+, webcam access, required dependencies installed locally, and optional DirectML-capable GPU.

## Latest Validation Result

Current validated machine:
- Python 3.11.9 in `.venv`
- Camera check passed on `camera_id=0`
- DirectML provider initialized but delivered about `0.12 FPS`
- CPU fallback delivered about `29.54 FPS` and is currently the practical runtime mode on this machine
