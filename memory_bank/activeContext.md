# Active Context: Posture Sentinel

## Current Focus

The product-level ergonomics pass is now implemented on top of the completed posture-quality work. The runtime now supports away/return awareness, break reminders, session analytics, more specific feedback reasons, and an additional operator-facing control for how strictly the app decides whether posture is truly straight or only borderline.

## Current Session Decisions

- Keep `docs/README.md` as the canonical architecture document for Memory Bank synchronization.
- Treat missing `mediapipe.solutions` as a supported runtime variant and fall back to manual preview-point rendering.
- Use bounded smoke execution for validation so automation can confirm successful startup without hanging indefinitely.
- Improve posture quality primarily through runtime robustness and operator UX rather than by adding new external dependencies.
- Keep the implementation local-first and config-driven so users can tune sensitivity without touching core code.
- Treat the new posture-quality pass as confirmed project scope and track it as a dedicated deliverable in `memory_bank/projectbrief.md`.
- Prefer stricter calibration quality gates over permissive baseline capture so the "good posture" state is more trustworthy after setup.
- Treat the new ergonomics-assistant layer as confirmed project scope and track it as a dedicated deliverable in `memory_bank/projectbrief.md`.
- Prefer user-facing usefulness over raw detector cleverness: reminders, away detection, and session context should reduce false pressure on the user.

## Verified Project State

- Core runtime, overlay, tray, headless mode, summaries, perf reporting, and threshold tuning are implemented.
- `.venv` bootstrap succeeded and all required runtime dependencies installed successfully.
- `tools/run_tests.ps1` passes, including environment diagnostics, `unittest`, and CLI smoke checks.
- `tools/smoke_test.ps1` now completes successfully with camera check, ONNX startup, and bounded shutdown.
- A bounded runtime produced real `logs/2026-03-24.perf.jsonl` data and `logs/2026-03-24.perf.summary.json`.
- On this validated machine, DirectML produced about `0.12 FPS` and auto-fell back to CPU, which then ran at about `29.54 FPS`.

## Immediate Blocker

No hard delivery blocker is present. The remaining work is now iterative tuning from real-world use rather than missing ergonomics-assistant functionality.

## Next Execution Steps

1. Collect real usage logs to tune the default straight-posture threshold and break-reminder cadence.
2. Review the new daily quality trends and see whether they are enough before adding heavier historical dashboards.
3. Consider whether away detection should also consider very low-confidence partial landmarks, not only full pose absence.
