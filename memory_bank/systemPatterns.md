# System Patterns: Posture Sentinel

## Runtime Architecture

- `main.py` owns application bootstrap, camera capture, provider selection, calibration state, violation detection, and desktop UX orchestration.
- Pose inference runs through ONNX Runtime with `DmlExecutionProvider` preferred and `CPUExecutionProvider` as explicit fallback.
- Posture decisions are threshold-based and compare current landmark metrics against a persisted baseline in `config.yaml`.
- Soft feedback is layered through preview rendering, tray state, and a click-through fullscreen overlay.
- Preview rendering falls back to manual landmark-point drawing when the installed `mediapipe` package does not expose the legacy `solutions` helpers.

## Data Flow Patterns

- Violation events are appended to daily JSONL files under `logs/`.
- Summary and performance CLIs rebuild derived reports from log files instead of depending on GUI session state.
- Threshold tuning reads recent log history, applies bounded adjustments, and writes updated runtime configuration locally.

## Resilience Patterns

- Camera reconnects are retried inside the runtime loop.
- Provider downgrade from DirectML to CPU happens when performance falls below configured thresholds.
- Overlay and tray initialization failures are handled explicitly to avoid silent process termination.
- Bounded smoke execution reuses the normal runtime path and stops cleanly after a configured interval, exporting summaries when available.

## Verification Patterns

- Non-UI behavior is covered by `unittest` tests in `tests/`.
- Runtime fallback behavior is verified through stubbed imports and monkeypatched objects.
- Environment readiness is checked before launch through `tools/check_env.py`.
- Bootstrap, run, test, and smoke automation are handled by PowerShell scripts in `tools/`.
- Smoke automation uses a bounded runtime window so startup validation can complete without manual interruption.
