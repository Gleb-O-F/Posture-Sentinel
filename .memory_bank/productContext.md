# Product Context: Posture Sentinel

## User Problem

People who work at a desk for long periods often slip into poor posture and do not notice it until discomfort accumulates.

## Product Value

Posture Sentinel gives local, privacy-preserving posture feedback without sending video to the cloud. The app is intended to stay lightweight, run in the background, and make correction noticeable without using aggressive interruptions.

## Current User Experience

1. User launches the app.
2. App checks environment and opens camera.
3. User calibrates a baseline posture.
4. App monitors posture in real time.
5. Violations trigger blur feedback and are logged locally.
6. Logs can be summarized and used to tune thresholds.

## Current Testing Goal

The immediate goal is to reach a machine-ready state where the project can be smoke-tested on a Windows workstation with a webcam and a working Python runtime.

## Constraints

- Local-only processing is required.
- Python runtime is currently missing or broken in the present environment.
- Final validation still depends on target hardware testing.
