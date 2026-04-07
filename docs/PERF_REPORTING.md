# Performance Reporting

Posture Sentinel writes runtime telemetry to `logs/YYYY-MM-DD.perf.jsonl`.

## Event Types

- `performance`: periodic FPS snapshot with provider, camera state, and runtime flags.
- `provider_switch`: records automatic provider downgrade, for example `dml -> cpu`.
- `overlay_failure`: emitted when the overlay cannot initialize.
- `tray_failure`: emitted when the system tray icon cannot initialize.

## CLI Usage

Generate a performance summary for one day:

```bash
python tools/perf_report.py --date 2026-03-13 --print
```

Generate a summary for a date range:

```bash
python tools/perf_report.py --from 2026-03-10 --to 2026-03-13 --print
```

Write summary to a custom file:

```bash
python tools/perf_report.py --date 2026-03-13 --out logs/2026-03-13.perf.summary.json
```

## Summary Output

The generated summary includes:

- `event_counts`
- `provider_counts`
- `camera_status_counts`
- `fps_by_provider` with `samples`, `avg_fps`, `min_fps`, `max_fps`
- `provider_switches`
- `failures`
- `malformed_lines`

## Recommended Profiling Flow

1. Run the app on the target machine long enough to produce `*.perf.jsonl` logs.
2. Build a daily or range summary with `tools/perf_report.py`.
3. Compare `avg_fps` and `provider_switches` before changing thresholds.
4. Tune `min_fps_before_cpu_fallback` only after reviewing real telemetry.

## Posture Quality Reporting

Violation logs now also contain `posture_state`, `posture_score`, and `tracking_score`.

Generate a quality summary for one day:

```bash
python tools/quality_report.py --date 2026-04-07 --print
```

Generate a quality summary for a date range:

```bash
python tools/quality_report.py --from 2026-04-05 --to 2026-04-07 --print
```

The quality summary includes:

- `posture_state_counts`
- `avg_posture_score`
- `avg_tracking_score`
- `max_posture_score`
- `min_tracking_score`
- `violation_quality`
- `recommendations`

The runtime performance log can also contain `quality_advice` events, which record when the app suggested recalibration, better framing, or sensitivity adjustments.
