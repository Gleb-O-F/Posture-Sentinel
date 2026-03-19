# Implementation Plan: Posture Sentinel

## Phase 0: Baseline Validation (Done)
- [x] Verify webcam access on target Windows machine
- [x] Validate pose model output for the required landmarks
- [x] Reach usable realtime posture detection

## Phase 1: Core Runtime (Done)
- [x] Implement config loading and persistence
- [x] Implement webcam capture and inference loop
- [x] Implement posture calibration and violation detection

## Phase 2: User Feedback (Done)
- [x] Add fullscreen blur overlay
- [x] Add preview window and tray status icon
- [x] Add manual calibration and preview toggle controls

## Phase 3: Reliability and Operations (Mostly done)
- [x] Add camera reconnect handling
- [x] Add DirectML to CPU fallback logic based on FPS
- [x] Add headless and no-tray modes
- [x] Add violation logging and daily summary export
- [x] Add threshold auto-tuning from local logs
- [x] Add startup smoke tests for non-UI modules
- [x] Add explicit handling for tray or overlay initialization failures

## Phase 4: Validation on Target Hardware (Next)
- [ ] Profile FPS and stability on the actual workstation GPU using `logs/*.perf.jsonl` and `tools/perf_report.py`
- [ ] Tune provider fallback thresholds from measured sessions
- [ ] Review violation and performance log quality after several days of use

## Phase 5: Product Direction (Decision)
- [ ] Decide whether Python remains the production runtime
- [ ] If Rust is still desired, create a separate migration plan instead of mixing it into current docs

---
Last updated: 2026-03-13
