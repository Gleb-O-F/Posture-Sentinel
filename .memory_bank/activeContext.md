# Active Context: Posture Sentinel

## Current Focus

Repository Git synchronization is configured for `https://github.com/fromptuo/Posture-Sentinel.git`, and the project remains prepared for immediate testing on a target Windows machine once a working Python 3.10+ interpreter is available.

## What Was Added Recently

- Git remote `origin` was repointed to `https://github.com/fromptuo/Posture-Sentinel.git`
- Branch `main` is synced and tracks `origin/main`
- Environment diagnostics: `environment_checks.py`, `tools/check_env.py`
- Performance telemetry summaries: `performance_reporting.py`, `tools/perf_report.py`
- Test automation scripts: `tools/bootstrap.ps1`, `tools/run.ps1`, `tools/run_tests.ps1`, `tools/smoke_test.ps1`
- Test documentation: `docs/PERF_REPORTING.md`, `docs/SETUP_AND_RUN.md`, `docs/TEST_PLAN.md`
- Unit tests for reporting, tuning, runtime fallback, perf reporting, perf CLI, and environment checks

## Immediate Blocker

The current environment still has no working Python runtime. This blocks dependency installation, unit test execution, CLI execution, and smoke testing.

## Next Execution Steps On A Healthy Machine

1. Run `tools/bootstrap.ps1`
2. Run `tools/run_tests.ps1`
3. Run `tools/smoke_test.ps1`
4. Review generated `logs/*.jsonl` and `logs/*.perf.jsonl`
5. Tune fallback thresholds from actual perf data
