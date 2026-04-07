import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from reporting import date_range_days, write_summary_file


def _load_quality_entries(log_files: Iterable[Path]) -> tuple[list[dict], int]:
    entries: list[dict] = []
    malformed_lines = 0

    for log_file in log_files:
        if not log_file.exists():
            continue
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    malformed_lines += 1

    return entries, malformed_lines


def _safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_quality_summary(entries: Iterable[dict], malformed_lines: int, label: str, source_files: list[str]) -> dict:
    total_events = 0
    posture_state_counts: Dict[str, int] = {}
    posture_scores: list[float] = []
    tracking_scores: list[float] = []
    per_violation_scores: Dict[str, dict] = {}
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    for item in entries:
        total_events += 1
        posture_state = str(item.get("posture_state", "unknown"))
        violation = str(item.get("violation", "unknown"))
        posture_score = _safe_float(item.get("posture_score"))
        tracking_score = _safe_float(item.get("tracking_score"))
        timestamp = item.get("timestamp")

        posture_state_counts[posture_state] = posture_state_counts.get(posture_state, 0) + 1
        posture_scores.append(posture_score)
        tracking_scores.append(tracking_score)

        if timestamp:
            first_ts = timestamp if first_ts is None else min(first_ts, timestamp)
            last_ts = timestamp if last_ts is None else max(last_ts, timestamp)

        violation_stats = per_violation_scores.setdefault(
            violation,
            {
                "count": 0,
                "total_posture_score": 0.0,
                "total_tracking_score": 0.0,
                "max_posture_score": 0.0,
            },
        )
        violation_stats["count"] += 1
        violation_stats["total_posture_score"] += posture_score
        violation_stats["total_tracking_score"] += tracking_score
        violation_stats["max_posture_score"] = max(violation_stats["max_posture_score"], posture_score)

    for stats in per_violation_scores.values():
        count = max(int(stats["count"]), 1)
        stats["avg_posture_score"] = round(stats["total_posture_score"] / count, 3)
        stats["avg_tracking_score"] = round(stats["total_tracking_score"] / count, 3)
        stats["max_posture_score"] = round(float(stats["max_posture_score"]), 3)
        del stats["total_posture_score"]
        del stats["total_tracking_score"]

    avg_posture_score = round(sum(posture_scores) / max(len(posture_scores), 1), 3) if posture_scores else 0.0
    avg_tracking_score = round(sum(tracking_scores) / max(len(tracking_scores), 1), 3) if tracking_scores else 0.0
    recommendations: list[str] = []

    pending_count = posture_state_counts.get("pending", 0)
    tracking_low_count = posture_state_counts.get("tracking_low", 0)
    bad_count = posture_state_counts.get("bad", 0)
    total_scored = max(total_events, 1)

    if avg_tracking_score < 0.65 or tracking_low_count / total_scored >= 0.25:
        recommendations.append(
            "Tracking quality is frequently low. Improve lighting, move closer to the camera, and keep shoulders visible."
        )
    if pending_count / total_scored >= 0.3:
        recommendations.append(
            "Too many borderline posture events. Recalibrate with a stable upright pose or lower posture sensitivity."
        )
    if bad_count > 0 and avg_posture_score < 0.95:
        recommendations.append(
            "Violations are often only slightly above the limit. Consider making the 'good posture' threshold a bit less strict."
        )
    if not recommendations:
        recommendations.append("Posture quality looks stable. Keep collecting logs before changing thresholds.")

    return {
        "label": label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_log_files": source_files,
        "total_events": total_events,
        "first_event_at": first_ts,
        "last_event_at": last_ts,
        "posture_state_counts": posture_state_counts,
        "avg_posture_score": avg_posture_score,
        "avg_tracking_score": avg_tracking_score,
        "max_posture_score": round(max(posture_scores), 3) if posture_scores else 0.0,
        "min_tracking_score": round(min(tracking_scores), 3) if tracking_scores else 0.0,
        "violation_quality": per_violation_scores,
        "recommendations": recommendations,
        "malformed_lines": malformed_lines,
    }


def build_daily_quality_summary(logs_dir: Path, day: str) -> Optional[dict]:
    log_file = logs_dir / f"{day}.jsonl"
    entries, malformed_lines = _load_quality_entries([log_file])
    if not entries and malformed_lines == 0:
        return None
    return build_quality_summary(entries, malformed_lines, day, [str(log_file)])


def build_range_quality_summary(logs_dir: Path, date_from: str, date_to: str) -> Optional[dict]:
    days = date_range_days(date_from, date_to)
    files = [logs_dir / f"{day}.jsonl" for day in days]
    entries, malformed_lines = _load_quality_entries(files)
    if not entries and malformed_lines == 0:
        return None
    label = f"{date_from}_to_{date_to}"
    return build_quality_summary(entries, malformed_lines, label, [str(p) for p in files if p.exists()])


def write_quality_summary_file(logs_dir: Path, filename: str, summary: dict) -> Path:
    return write_summary_file(logs_dir, filename, summary)
