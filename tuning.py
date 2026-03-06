import json
import math
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict


VIOLATION_TO_THRESHOLD = {
    "slouching": "slouch_threshold",
    "leaning_forward": "lean_threshold",
    "shoulder_tilt": "shoulder_tilt_threshold",
    "body_rotation": "shoulder_width_threshold",
}

THRESHOLD_BOUNDS = {
    "slouch_threshold": (0.03, 0.30),
    "lean_threshold": (0.05, 0.35),
    "shoulder_tilt_threshold": (0.03, 0.25),
    "shoulder_width_threshold": (0.55, 0.92),
}


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _collect_entries(logs_dir: Path, lookback_days: int) -> tuple[list[dict], int, int]:
    existing_days = 0
    malformed_lines = 0
    entries: list[dict] = []
    today = datetime.now().date()

    for offset in range(lookback_days):
        day = (today - timedelta(days=offset)).isoformat()
        log_file = logs_dir / f"{day}.jsonl"
        if not log_file.exists():
            continue
        existing_days += 1
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed_lines += 1

    return entries, existing_days, malformed_lines


def tune_thresholds(config_obj, logs_dir: Path, lookback_days: int = 7, target_events_per_day: float = 12.0) -> dict:
    if lookback_days < 3:
        lookback_days = 3

    entries, existing_days, malformed_lines = _collect_entries(logs_dir, lookback_days)
    if existing_days < 3:
        return {
            "applied": False,
            "reason": f"insufficient_history_days:{existing_days}",
            "lookback_days": lookback_days,
            "history_days_with_logs": existing_days,
            "malformed_lines": malformed_lines,
        }

    per_violation_count: Dict[str, int] = {}
    for item in entries:
        v = str(item.get("violation", "unknown"))
        per_violation_count[v] = per_violation_count.get(v, 0) + 1

    total_events = sum(per_violation_count.get(k, 0) for k in VIOLATION_TO_THRESHOLD.keys())
    if total_events < 15:
        return {
            "applied": False,
            "reason": f"insufficient_events:{total_events}",
            "lookback_days": lookback_days,
            "history_days_with_logs": existing_days,
            "malformed_lines": malformed_lines,
            "events": per_violation_count,
        }

    if is_dataclass(config_obj):
        cfg_before = asdict(config_obj)
    else:
        cfg_before = dict(getattr(config_obj, "__dict__", {}))
    cfg_after = dict(cfg_before)
    changes: Dict[str, dict] = {}

    for violation, threshold_name in VIOLATION_TO_THRESHOLD.items():
        if threshold_name not in cfg_before:
            continue
        old_value = float(cfg_before[threshold_name])
        events = per_violation_count.get(violation, 0)
        avg_per_day = events / max(existing_days, 1)
        ratio = avg_per_day / max(target_events_per_day, 1e-6)

        # Conservative logarithmic adjustment, capped to +/-20% per run.
        raw_adj = math.log2(max(ratio, 1e-6)) * 0.15
        adj = _clamp(raw_adj, -0.20, 0.20)

        # For width threshold, lower value means less sensitivity (inverse behavior).
        if threshold_name == "shoulder_width_threshold":
            new_value = old_value * (1.0 - adj)
        else:
            new_value = old_value * (1.0 + adj)

        lo, hi = THRESHOLD_BOUNDS[threshold_name]
        new_value = _clamp(new_value, lo, hi)

        # Ignore tiny updates.
        if abs(new_value - old_value) < 0.003:
            continue

        cfg_after[threshold_name] = round(new_value, 4)
        changes[threshold_name] = {
            "old": round(old_value, 4),
            "new": round(new_value, 4),
            "violation": violation,
            "avg_events_per_day": round(avg_per_day, 2),
        }

    if not changes:
        return {
            "applied": False,
            "reason": "no_material_change",
            "lookback_days": lookback_days,
            "history_days_with_logs": existing_days,
            "malformed_lines": malformed_lines,
            "events": per_violation_count,
        }

    for k, v in cfg_after.items():
        setattr(config_obj, k, v)

    report = {
        "applied": True,
        "lookback_days": lookback_days,
        "history_days_with_logs": existing_days,
        "malformed_lines": malformed_lines,
        "target_events_per_day": target_events_per_day,
        "events": per_violation_count,
        "changes": changes,
        "tuned_at": datetime.now().isoformat(timespec="seconds"),
    }
    return report
