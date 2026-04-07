import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Optional

from reporting import date_range_days, write_summary_file


def _load_perf_entries(log_files: Iterable[Path]) -> tuple[list[dict], int]:
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


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_daily_session_trends(session_summaries: Iterable[dict]) -> list[dict]:
    by_day: Dict[str, dict] = {}

    for item in session_summaries:
        timestamp = str(item.get("timestamp", ""))
        day = timestamp.split("T", 1)[0] if "T" in timestamp else timestamp[:10]
        if not day:
            day = "unknown"

        trend = by_day.setdefault(
            day,
            {
                "day": day,
                "sessions": 0,
                "total_session_minutes": 0.0,
                "total_present_minutes": 0.0,
                "total_away_count": 0,
                "total_return_count": 0,
                "total_break_reminders": 0,
            },
        )
        trend["sessions"] += 1
        trend["total_session_minutes"] += _safe_float(item.get("session_minutes"))
        trend["total_present_minutes"] += _safe_float(item.get("present_minutes"))
        trend["total_away_count"] += _safe_int(item.get("away_count"))
        trend["total_return_count"] += _safe_int(item.get("return_count"))
        trend["total_break_reminders"] += _safe_int(item.get("break_reminders"))

    daily_trends = []
    for day in sorted(by_day.keys()):
        trend = by_day[day]
        sessions = max(int(trend["sessions"]), 1)
        daily_trends.append(
            {
                "day": day,
                "sessions": int(trend["sessions"]),
                "avg_session_minutes": round(trend["total_session_minutes"] / sessions, 2),
                "avg_present_minutes": round(trend["total_present_minutes"] / sessions, 2),
                "total_away_count": int(trend["total_away_count"]),
                "total_return_count": int(trend["total_return_count"]),
                "total_break_reminders": int(trend["total_break_reminders"]),
            }
        )

    return daily_trends


def build_perf_summary(entries: Iterable[dict], malformed_lines: int, label: str, source_files: list[str]) -> dict:
    total_events = 0
    event_counts: Dict[str, int] = {}
    provider_counts: Dict[str, int] = {}
    camera_status_counts: Dict[str, int] = {}
    fps_by_provider: Dict[str, dict] = {}
    provider_switches: list[dict] = []
    failures: list[dict] = []
    quality_advice_events: list[dict] = []
    session_summaries: list[dict] = []
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    for item in entries:
        total_events += 1
        event_type = str(item.get("event", "unknown"))
        provider = str(item.get("provider", "unknown"))
        camera_status = str(item.get("camera_status", "unknown"))
        timestamp = item.get("timestamp")

        event_counts[event_type] = event_counts.get(event_type, 0) + 1
        provider_counts[provider] = provider_counts.get(provider, 0) + 1
        camera_status_counts[camera_status] = camera_status_counts.get(camera_status, 0) + 1

        if timestamp:
            first_ts = timestamp if first_ts is None else min(first_ts, timestamp)
            last_ts = timestamp if last_ts is None else max(last_ts, timestamp)

        if event_type == "performance":
            try:
                fps = float(item.get("fps", 0.0))
            except (TypeError, ValueError):
                fps = 0.0
            stats = fps_by_provider.setdefault(provider, {"samples": 0, "min_fps": None, "max_fps": None, "total_fps": 0.0})
            stats["samples"] += 1
            stats["total_fps"] += fps
            stats["min_fps"] = fps if stats["min_fps"] is None else min(stats["min_fps"], fps)
            stats["max_fps"] = fps if stats["max_fps"] is None else max(stats["max_fps"], fps)

        if event_type == "provider_switch":
            provider_switches.append(
                {
                    "timestamp": timestamp,
                    "previous_provider": item.get("previous_provider"),
                    "new_provider": item.get("new_provider"),
                    "reason": item.get("reason"),
                    "fps": item.get("fps"),
                }
            )

        if event_type.endswith("_failure"):
            failures.append(
                {
                    "timestamp": timestamp,
                    "event": event_type,
                    "provider": provider,
                    "error": item.get("error"),
                }
            )

        if event_type == "quality_advice":
            quality_advice_events.append(
                {
                    "timestamp": timestamp,
                    "advice": item.get("advice"),
                }
            )

        if event_type == "session_summary":
            session_summaries.append(
                {
                    "timestamp": timestamp,
                    "session_minutes": item.get("session_minutes"),
                    "present_minutes": item.get("present_minutes"),
                    "away_count": item.get("away_count"),
                    "return_count": item.get("return_count"),
                    "break_reminders": item.get("break_reminders"),
                    "state_minutes": item.get("state_minutes"),
                }
            )

    fps_summary: Dict[str, dict] = {}
    for provider, stats in fps_by_provider.items():
        samples = int(stats["samples"])
        avg_fps = stats["total_fps"] / max(samples, 1)
        fps_summary[provider] = {
            "samples": samples,
            "avg_fps": round(avg_fps, 2),
            "min_fps": round(float(stats["min_fps"]), 2) if stats["min_fps"] is not None else None,
            "max_fps": round(float(stats["max_fps"]), 2) if stats["max_fps"] is not None else None,
        }

    return {
        "label": label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_perf_files": source_files,
        "total_events": total_events,
        "first_event_at": first_ts,
        "last_event_at": last_ts,
        "event_counts": event_counts,
        "provider_counts": provider_counts,
        "camera_status_counts": camera_status_counts,
        "fps_by_provider": fps_summary,
        "provider_switches": provider_switches,
        "failures": failures,
        "quality_advice_events": quality_advice_events,
        "session_summaries": session_summaries,
        "daily_session_trends": _build_daily_session_trends(session_summaries),
        "malformed_lines": malformed_lines,
    }


def build_daily_perf_summary(logs_dir: Path, day: str) -> Optional[dict]:
    log_file = logs_dir / f"{day}.perf.jsonl"
    entries, malformed_lines = _load_perf_entries([log_file])
    if not entries and malformed_lines == 0:
        return None
    return build_perf_summary(entries, malformed_lines, day, [str(log_file)])


def build_range_perf_summary(logs_dir: Path, date_from: str, date_to: str) -> Optional[dict]:
    days = date_range_days(date_from, date_to)
    files = [logs_dir / f"{day}.perf.jsonl" for day in days]
    entries, malformed_lines = _load_perf_entries(files)
    if not entries and malformed_lines == 0:
        return None
    label = f"{date_from}_to_{date_to}"
    return build_perf_summary(entries, malformed_lines, label, [str(p) for p in files if p.exists()])


def write_perf_summary_file(logs_dir: Path, filename: str, summary: dict) -> Path:
    return write_summary_file(logs_dir, filename, summary)
