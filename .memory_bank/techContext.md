# Tech Context: Posture Sentinel

## Main Stack

- Python 3.10+
- OpenCV
- MediaPipe drawing helpers
- ONNX Runtime / ONNX Runtime DirectML
- NumPy
- Pillow
- pystray
- pywin32
- Tkinter

## Important Files

- `main.py`: main runtime and UX orchestration
- `reporting.py`: violation log aggregation
- `tuning.py`: threshold auto-tuning logic
- `performance_reporting.py`: perf log aggregation
- `tools/check_env.py`: environment diagnostics CLI
- `tools/report.py`: violation summary CLI
- `tools/perf_report.py`: performance summary CLI
- `tools/tune.py`: tuning CLI
- `tools/bootstrap.ps1`: environment bootstrap
- `tools/run.ps1`: guarded app launch
- `tools/run_tests.ps1`: automated test pass
- `tools/smoke_test.ps1`: smoke-test runner

## Current Environment Constraint

Repository structure is ready, but the current workspace still lacks a working Python interpreter for real execution.

## Validation Target

A Windows machine with Python 3.10+, webcam access, and required dependencies installed in a local virtual environment.
