# System Patterns: Posture Sentinel

## Runtime Architecture

- Camera capture and frame loop live in `main.py`
- Landmark inference uses ONNX Runtime with DirectML preferred and CPU fallback
- Posture analysis uses a calibrated baseline and threshold-based violation logic
- User feedback uses preview rendering, tray state, and a fullscreen overlay

## Operational Patterns

- Posture events are stored in daily JSONL files under `logs/`
- Performance events are stored in daily `.perf.jsonl` files under `logs/`
- Reporting CLIs build summaries from log files instead of depending on a running GUI session
- Runtime degradation is explicit: overlay and tray failures are caught and logged instead of crashing the app silently

## Verification Patterns

- Non-UI logic is covered with `unittest`
- GUI fallback behavior is tested through stubbed imports and monkeypatched runtime objects
- Environment readiness is checked before the app is launched through `tools/check_env.py`
- Setup and smoke-test automation is handled by PowerShell scripts in `tools/`
