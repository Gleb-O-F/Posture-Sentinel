# Active Context: Posture Sentinel

## Current Focus

Memory Bank is being synchronized with the current `memory_bank/` path, the architectural source of truth is being restored in `docs/README.md`, and the repository is being prepared for a full commit and push.

## Current Session Decisions

- Use the upstream `AGENTS.md` from `Ravva/projects-tracker` as the local rule set.
- Treat `docs/README.md` as the canonical architecture document and align `memory_bank` to it.
- Preserve the existing project scope: Python runtime is current production, while final validation remains blocked by lack of a working Python runtime on the target machine.

## Verified Project State

- Core runtime, overlay, tray, headless mode, summaries, perf reporting, and threshold tuning are implemented.
- PowerShell bootstrap, run, test, and smoke automation are present under `tools/`.
- Unit tests and diagnostic tooling exist, but target-hardware validation is still outstanding.
- Git history currently contains an unfinished migration from `.memory_bank/` to `memory_bank/`.

## Immediate Blocker

Successful push still depends on GitHub authentication with write access to `fromptuo/Posture-Sentinel`.

## Next Execution Steps

1. Commit the synchronized documentation state, including the `memory_bank/` path.
2. Push the repository to `origin/main`.
3. On a healthy Windows machine, run `tools/bootstrap.ps1`, `tools/run_tests.ps1`, and `tools/smoke_test.ps1`.
4. Review generated runtime and performance logs, then tune thresholds from real sessions.
