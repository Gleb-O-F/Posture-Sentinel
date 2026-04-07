# Project Brief: Posture Sentinel

## Overview

Posture Sentinel is a Windows desktop application for local real-time posture monitoring with soft corrective feedback, local analytics, and operational tooling for validation on target hardware.

## Scope Baseline

The current production scope is the Python implementation described in `docs/README.md`. The project is considered complete only after the documented runtime, operational tooling, and target-machine validation deliverables are closed.

## Target Platform

- Windows 10/11
- Python 3.10+
- Local webcam access
- Optional DirectML-capable GPU for better inference throughput

## Project Deliverables

| ID | Deliverable | Status | Weight |
|----|-------------|--------|--------|
| PS-01 | Core runtime for webcam capture, ONNX inference, posture calibration, and violation detection | completed | 14 |
| PS-02 | User feedback layer with preview UI, tray integration, overlay, headless mode, and no-tray mode | completed | 12 |
| PS-03 | Local analytics and tuning with violation logs, summaries, perf logs, and tuning/reporting CLIs | completed | 15 |
| PS-04 | Automated verification and diagnostics with unit tests, fallback coverage, and environment checks | completed | 15 |
| PS-05 | Windows setup and execution automation with bootstrap, run, test, and smoke scripts plus operator docs | completed | 8 |
| PS-06 | End-to-end validation on target hardware with working Python runtime, dependency install, smoke run, and perf review | completed | 11 |
| PS-07 | Posture quality pass with smoothing, hysteresis, calibration quality gates, posture accuracy tuning, and richer analytics | completed | 13 |
| PS-08 | Ergonomics assistant features with away detection, break reminders, session analytics, reason-specific feedback, and stricter straight-posture tuning | completed | 12 |

## Completion Rule

Project completion percentage must be calculated only from the deliverables table above.
Current weighted completion: 100% completed, 0% in_progress, 0% pending.

## Traceability

- Architecture source of truth: `docs/README.md`
- Product requirements: `docs/PRD.md`
- Delivery plan: `docs/IMPLEMENTATION_PLAN.md`
- Operating docs: `docs/PERF_REPORTING.md`, `docs/SETUP_AND_RUN.md`, `docs/TEST_PLAN.md`
