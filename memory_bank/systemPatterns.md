# System Patterns: Posture Sentinel

## Runtime Architecture

- `main.py` owns application bootstrap, camera capture, provider selection, calibration state, violation detection, and desktop UX orchestration.
- Pose inference runs through ONNX Runtime with `DmlExecutionProvider` preferred and `CPUExecutionProvider` as explicit fallback.
- Posture decisions compare smoothed landmark metrics against a persisted baseline in `config.yaml`, then use posture-score thresholds plus hysteresis to stabilize transitions between `good`, `pending`, `bad`, `tracking_low`, and `away`.
- Soft feedback is layered through preview rendering, tray state, and a click-through fullscreen overlay.
- Preview rendering falls back to manual landmark-point drawing when the installed `mediapipe` package does not expose the legacy `solutions` helpers.
- Calibration now depends on minimum tracking quality so low-confidence frames do not establish a misleading baseline.
- Broken or truncated `config.yaml` files are backed up and replaced with a recoverable default config so the app can still start safely.
- Away detection pauses posture pressure when no person is reliably present, and a session timer tracks away/return transitions plus break reminders.

## Data Flow Patterns

- Violation events are appended to daily JSONL files under `logs/`.
- Runtime logs include posture score, tracking score, and posture state so misclassification can be analyzed after real usage.
- Summary and performance CLIs rebuild derived reports from log files instead of depending on GUI session state.
- A dedicated posture-quality CLI summarizes state distribution and score quality from existing violation logs without needing separate runtime instrumentation.
- The tray menu now doubles as a lightweight operator console for quality tuning and advice toggling, avoiding direct edits to `config.yaml` during normal use.
- Runtime quality advice is also written to performance telemetry so later analysis can compare recommendations against operator outcomes.
- Session summaries are emitted into performance telemetry on shutdown so later analysis can review time at desk, away counts, break reminders, and state durations.
- Posture-quality summaries now include per-day trend slices so tuning decisions can be based on movement over time, not only one aggregate snapshot.
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
