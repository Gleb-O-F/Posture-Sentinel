# Tech Context: Posture Sentinel

## Main Stack

- Python 3.10+
- OpenCV
- MediaPipe
- NumPy
- ONNX Runtime / ONNX Runtime DirectML
- Pillow
- pystray
- pywin32
- Tkinter

## Important Files

- `main.py`: runtime, inference, calibration, UI orchestration, logging
- `reporting.py`: violation aggregation and summary serialization
- `performance_reporting.py`: performance telemetry aggregation
- `tuning.py`: bounded threshold auto-tuning
- `tools/check_env.py`: environment diagnostics
- `tools/report.py`: violation summary CLI
- `tools/perf_report.py`: performance summary CLI
- `tools/tune.py`: tuning CLI
- `tools/bootstrap.ps1`: environment bootstrap
- `tools/run.ps1`: guarded launch
- `tools/run_tests.ps1`: automated tests
- `tools/smoke_test.ps1`: smoke run

## Dependencies Source

Required Python packages are declared in `requirements.txt`.

## Current Environment Constraint

The repository is structurally ready, but the current workspace still lacks a working Python interpreter for real execution and smoke validation.

## Validation Target

A Windows workstation with Python 3.10+, webcam access, required dependencies installed locally, and optional DirectML-capable GPU.
