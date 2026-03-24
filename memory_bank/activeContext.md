# Active Context: Posture Sentinel

## Current Focus

The project has now completed local target-hardware validation on the current Windows machine. The active focus is no longer implementation readiness, but optional follow-up optimization from longer real-world usage data.

## Current Session Decisions

- Keep `docs/README.md` as the canonical architecture document for Memory Bank synchronization.
- Treat missing `mediapipe.solutions` as a supported runtime variant and fall back to manual preview-point rendering.
- Use bounded smoke execution for validation so automation can confirm successful startup without hanging indefinitely.

## Verified Project State

- Core runtime, overlay, tray, headless mode, summaries, perf reporting, and threshold tuning are implemented.
- `.venv` bootstrap succeeded and all required runtime dependencies installed successfully.
- `tools/run_tests.ps1` passes, including environment diagnostics, `unittest`, and CLI smoke checks.
- `tools/smoke_test.ps1` now completes successfully with camera check, ONNX startup, and bounded shutdown.
- A bounded runtime produced real `logs/2026-03-24.perf.jsonl` data and `logs/2026-03-24.perf.summary.json`.
- On this validated machine, DirectML produced about `0.12 FPS` and auto-fell back to CPU, which then ran at about `29.54 FPS`.

## Immediate Blocker

No delivery blocker remains. Only optional product-level tuning from longer real usage sessions is left.

## Next Execution Steps

1. Optionally collect multi-day violation logs from real use.
2. Re-run `tools/report.py`, `tools/perf_report.py`, and `tools/tune.py` after more history accumulates.
3. If CPU remains the only viable provider on this machine, adjust defaults or operator guidance accordingly.
