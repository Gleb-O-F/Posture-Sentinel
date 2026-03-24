# Progress: Posture Sentinel

## Current State

Memory Bank is synchronized, the local Windows environment is operational, and the project now passes bootstrap, diagnostics, automated tests, bounded smoke validation, and a real perf-log review on the target machine. Project deliverables are functionally complete.

## Completed Areas

- Python runtime for posture monitoring exists
- Tray, overlay, headless, and no-tray modes exist
- Violation logging and daily summaries exist
- Performance telemetry and perf summary tooling exist
- Threshold auto-tuning logic and CLI exist
- Environment diagnostics exist
- Bootstrap, run, test, and smoke PowerShell scripts exist
- Memory Bank structure has been repaired under `memory_bank/`
- `Project Deliverables` now remain explicitly normalized to a markdown table with total weight `100`
- Local `.venv` bootstrap completed successfully
- `tools/run_tests.ps1` passes in the managed environment
- `tools/smoke_test.ps1` passes with camera access and ONNX startup
- A bounded real run produced performance telemetry and a generated perf summary
- Target-machine validation is complete according to the current deliverables table

## Known Issues

- DirectML is technically available on this machine but performs poorly here (`~0.12 FPS`) and immediately falls back to CPU
- CPU fallback is the practical runtime path on the validated machine (`~29.54 FPS` in the recorded run)
- Daily summary export correctly skips output when no violation log exists yet, so the smoke pass alone does not generate analytics artifacts
- `pip show` in the current console can emit cp1251 encoding errors for package metadata containing non-ASCII characters, although installation and runtime are unaffected

## Changelog

| Date | Change |
|------|--------|
| 2026-03-13 | Added startup hardening, runtime fallback tests, perf reporting, setup/test automation docs and scripts |
| 2026-03-19 | Added environment diagnostics, bootstrap/run/test/smoke PowerShell scripts, perf CLI coverage, and updated Memory Bank |
| 2026-03-20 | Reconfigured Git remote `origin` to `https://github.com/fromptuo/Posture-Sentinel.git`, verified fetch/push, and updated Memory Bank |
| 2026-03-24 | Replaced the local AGENTS file with the upstream version, restored `docs/README.md`, and resynchronized `memory_bank/` with current project state |
| 2026-03-24 | Fixed runtime-safe logging, made tuning skip empty-signal violations, added bounded smoke execution, validated camera/ONNX startup, and updated project progress to 85% |
| 2026-03-24 | Collected real perf telemetry on the validated machine, confirmed DirectML auto-fallback to CPU, generated `logs/2026-03-24.perf.summary.json`, and closed project completion to 100% |
| 2026-03-24 | Added a large `НЕРОВНАЯ ОСАНКА` warning to the overlay and preview when a posture violation is confirmed |

| 2026-03-24 | Added explicit `good/pending/bad` posture feedback states so the preview now visibly indicates when posture is straight |

## Change Control

last_checked_commit: f66db1c8d4abf5b26469b080c25de0128a472acf
