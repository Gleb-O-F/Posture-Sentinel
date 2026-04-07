# Posture Sentinel Architecture

## Overview

Posture Sentinel is a Windows desktop application for local real-time posture monitoring from a webcam feed. The production implementation is Python-based and prioritizes local inference, soft corrective feedback, and operational tooling for validation on target hardware.

## Architecture

### Runtime flow

1. `main.py` loads persisted settings from `config.yaml`.
2. The app opens a webcam stream and runs ONNX pose inference with DirectML preferred and CPU fallback.
3. Landmark-derived metrics are smoothed, checked for tracking quality, and compared against a saved baseline and configured thresholds.
4. A posture score and hysteresis-driven state machine distinguish `good`, `pending`, `bad`, `tracking_low`, and `uncalibrated` states, using an explicit posture-accuracy threshold to better recognize straight posture.
5. Confirmed violations raise user feedback through the preview UI, tray state, a fullscreen blur overlay, and a large warning label `НЕРОВНАЯ ОСАНКА`.
6. The tray menu exposes quick quality controls for posture sensitivity, straight-posture strictness, and quality-advice toggling.
7. Runtime events are logged locally for later summaries and threshold tuning, including richer quality and posture-state metadata.

### Main modules

| Module | Responsibility |
|--------|----------------|
| `main.py` | Application bootstrap, camera loop, ONNX session management, posture calibration, violation detection, tray and overlay orchestration |
| `reporting.py` | Aggregates daily and date-range violation logs into JSON summaries |
| `performance_reporting.py` | Aggregates performance telemetry, provider switches, FPS, and failures |
| `posture_quality_reporting.py` | Aggregates posture score, tracking score, and posture-state quality summaries from violation logs |
| `tuning.py` | Tunes posture thresholds from recent local history with bounded adjustments |
| `tools/check_env.py` | Verifies local environment readiness before launch |
| `tools/report.py` | CLI entry point for violation summaries |
| `tools/perf_report.py` | CLI entry point for performance summaries |
| `tools/quality_report.py` | CLI entry point for posture-quality summaries and recommendation output |
| `tools/tune.py` | CLI entry point for threshold tuning |
| `tools/*.ps1` | Bootstrap, launch, test, and smoke-test automation for Windows |

## Data and persistence

- Runtime configuration is stored in `config.yaml`.
- Violation events are written to `logs/YYYY-MM-DD.jsonl`.
- Daily violation summaries are written to `logs/YYYY-MM-DD.summary.json`.
- Performance telemetry is written to `logs/YYYY-MM-DD.perf.jsonl`.
- Calibration quality, posture score, tracking score, and posture-state metadata are persisted in runtime logs for diagnostics.

## Verification and operations

- Automated verification is centered on `unittest` coverage in `tests/`.
- Smoke and environment automation are handled by PowerShell scripts in `tools/`.
- Final production validation still depends on a Windows machine with a working Python 3.10+ runtime, webcam access, and optional DirectML-capable GPU.

## Scope boundaries

- The current source of truth is the Python implementation; older Rust planning artifacts are historical only.
- The system is local-first and does not send video to the cloud.
- Project completion must be derived from `memory_bank/projectbrief.md`, while this file remains the architectural source of truth.
