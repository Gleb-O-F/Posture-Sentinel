# Active Context: Posture Sentinel

## Current Focus

The project has completed the active quality pass over posture recognition UX on the current Windows machine. The runtime now includes metric smoothing and hysteresis for stable classification, calibration quality gates, tracking-quality feedback, posture sensitivity settings, enriched analytics, and an explicit posture-accuracy control so the app better distinguishes straight posture from borderline poses.

## Current Session Decisions

- Keep `docs/README.md` as the canonical architecture document for Memory Bank synchronization.
- Treat missing `mediapipe.solutions` as a supported runtime variant and fall back to manual preview-point rendering.
- Use bounded smoke execution for validation so automation can confirm successful startup without hanging indefinitely.
- Improve posture quality primarily through runtime robustness and operator UX rather than by adding new external dependencies.
- Keep the implementation local-first and config-driven so users can tune sensitivity without touching core code.
- Treat the new posture-quality pass as confirmed project scope and track it as a dedicated deliverable in `memory_bank/projectbrief.md`.
- Prefer stricter calibration quality gates over permissive baseline capture so the "good posture" state is more trustworthy after setup.

## Verified Project State

- Core runtime, overlay, tray, headless mode, summaries, perf reporting, and threshold tuning are implemented.
- `.venv` bootstrap succeeded and all required runtime dependencies installed successfully.
- `tools/run_tests.ps1` passes, including environment diagnostics, `unittest`, and CLI smoke checks.
- `tools/smoke_test.ps1` now completes successfully with camera check, ONNX startup, and bounded shutdown.
- A bounded runtime produced real `logs/2026-03-24.perf.jsonl` data and `logs/2026-03-24.perf.summary.json`.
- On this validated machine, DirectML produced about `0.12 FPS` and auto-fell back to CPU, which then ran at about `29.54 FPS`.

## Immediate Blocker

No delivery blocker is currently present. The remaining work is optional tuning from longer real-world usage sessions rather than missing implementation.

## Next Execution Steps

1. Collect several days of real usage logs with the new posture score and tracking score fields.
2. Review the new `tools/quality_report.py` output and runtime quality advice to see whether the default sensitivity and good-posture threshold need machine-specific tuning.
3. If users still see borderline misclassification, refine the posture score formula using the richer telemetry now written to logs.
