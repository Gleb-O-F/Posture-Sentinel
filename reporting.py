import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional


def date_range_days(date_from: str, date_to: str) -> list[str]:
    start = datetime.strptime(date_from, "%Y-%m-%d").date()
    end = datetime.strptime(date_to, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("date_to must be >= date_from")
    days = []
    current = start
    while current <= end:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def _load_log_entries(log_files: Iterable[Path]) -> tuple[list[dict], int]:
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


def build_summary(entries: Iterable[dict], malformed_lines: int, label: str, source_files: list[str]) -> dict:
    total_events = 0
    total_duration_sec = 0.0
    per_violation: Dict[str, Dict[str, float]] = {}
    camera_status_counts: Dict[str, int] = {}
    first_ts: Optional[str] = None
    last_ts: Optional[str] = None

    for item in entries:
        violation = str(item.get("violation", "unknown"))
        camera_status = str(item.get("camera_status", "unknown"))
        timestamp = item.get("timestamp")
        try:
            duration_sec = float(item.get("duration_sec", 0.0))
        except (TypeError, ValueError):
            duration_sec = 0.0

        total_events += 1
        total_duration_sec += duration_sec
        if timestamp:
            first_ts = timestamp if first_ts is None else min(first_ts, timestamp)
            last_ts = timestamp if last_ts is None else max(last_ts, timestamp)

        if violation not in per_violation:
            per_violation[violation] = {"count": 0, "total_duration_sec": 0.0}
        per_violation[violation]["count"] += 1
        per_violation[violation]["total_duration_sec"] += duration_sec

        camera_status_counts[camera_status] = camera_status_counts.get(camera_status, 0) + 1

    for _, stats in per_violation.items():
        count = int(stats["count"])
        total_dur = float(stats["total_duration_sec"])
        stats["avg_duration_sec"] = round(total_dur / max(count, 1), 2)
        stats["total_duration_sec"] = round(total_dur, 2)

    return {
        "label": label,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_log_files": source_files,
        "total_events": total_events,
        "total_duration_sec": round(total_duration_sec, 2),
        "first_event_at": first_ts,
        "last_event_at": last_ts,
        "camera_status_counts": camera_status_counts,
        "violations": per_violation,
        "malformed_lines": malformed_lines,
    }


def build_daily_summary(logs_dir: Path, day: str) -> Optional[dict]:
    log_file = logs_dir / f"{day}.jsonl"
    entries, malformed_lines = _load_log_entries([log_file])
    if not entries and malformed_lines == 0:
        return None
    return build_summary(entries, malformed_lines, day, [str(log_file)])


def build_range_summary(logs_dir: Path, date_from: str, date_to: str) -> Optional[dict]:
    days = date_range_days(date_from, date_to)
    files = [logs_dir / f"{day}.jsonl" for day in days]
    entries, malformed_lines = _load_log_entries(files)
    if not entries and malformed_lines == 0:
        return None
    label = f"{date_from}_to_{date_to}"
    return build_summary(entries, malformed_lines, label, [str(p) for p in files if p.exists()])


def write_summary_file(logs_dir: Path, filename: str, summary: dict) -> Path:
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_path = logs_dir / filename
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return output_path
