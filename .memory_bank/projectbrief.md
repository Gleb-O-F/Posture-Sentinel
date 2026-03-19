# Project Brief: Posture Sentinel

## Overview

Posture Sentinel is a Windows desktop application that monitors user posture in real time from a webcam feed and applies soft feedback when posture violations persist.

## Current Product Direction

The active implementation is Python-based and uses ONNX Runtime with DirectML preferred and CPU fallback. Earlier Rust planning artifacts are historical only and no longer describe the working codebase.

## Target Platform

- Windows 10/11
- Local webcam
- Python 3.10+
- Optional DirectML-capable GPU for better inference throughput

## Project Deliverables

| ID | Deliverable | Status | Weight |
|----|-------------|--------|--------|
| D1 | Core posture monitoring runtime (camera loop, ONNX inference, calibration, violation detection) | completed | 20 |
| D2 | User feedback and background UX (preview, tray, overlay, headless and no-tray modes) | completed | 15 |
| D3 | Local analytics and tuning (violation logs, summaries, perf logs, auto-tuning, reporting CLIs) | completed | 20 |
| D4 | Automated verification and diagnostics (unit tests, runtime fallback tests, environment check CLI) | in_progress | 20 |
| D5 | Setup and test automation for target machine (bootstrap, run, smoke-test, test scripts, docs) | in_progress | 10 |
| D6 | End-to-end validation on target hardware (working Python runtime, dependency install, smoke run, perf review) | pending | 15 |

## Completion Rule

Project completion percentage must be calculated only from the deliverables table above.
Current weighted completion: 65% completed, 30% in progress, 15% pending.

## Source Of Truth

- Runtime code: `main.py`, `reporting.py`, `tuning.py`, `performance_reporting.py`
- Product docs: `docs/PRD.md`, `docs/IMPLEMENTATION_PLAN.md`, `docs/PERF_REPORTING.md`, `docs/SETUP_AND_RUN.md`, `docs/TEST_PLAN.md`
