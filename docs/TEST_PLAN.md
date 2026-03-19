# Test Plan

## Goal

Reach a state where Posture Sentinel can be tested on a target Windows machine with a webcam and Python 3.10+.

## Preconditions

- Python 3.10+ is installed and working from `python` or `py -3`
- webcam is available
- ONNX model files exist in `models/`

## Automated Checks

Bootstrap and install dependencies:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\bootstrap.ps1
```

Run the automated test pass:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\run_tests.ps1
```

This covers:

- environment diagnostics
- unit tests in `tests/`
- CLI smoke checks for `report.py`, `perf_report.py`, and `tune.py`

## Smoke Test

Run a headless smoke session with camera probing:

```powershell
powershell -ExecutionPolicy Bypass -File .\tools\smoke_test.ps1
```

## Manual Validation

1. Run `powershell -ExecutionPolicy Bypass -File .\tools\run.ps1`.
2. Verify the camera opens and the app stays alive.
3. Calibrate once and confirm `config.yaml` updates.
4. Confirm tray and overlay behavior if not using `-NoTray`.
5. Let the app run long enough to create `logs/*.jsonl` and `logs/*.perf.jsonl`.
6. Build summaries with `tools/report.py` and `tools/perf_report.py`.
7. Review provider changes, FPS, and posture event frequency.

## Exit Criteria

The project is ready for broader testing when:

- `tools/run_tests.ps1` completes without unexpected failures
- `tools/smoke_test.ps1` starts successfully on the target machine
- the app records posture logs and perf logs
- no immediate crashes occur during tray, overlay, or camera startup
