import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tuning import tune_thresholds  # noqa: E402


DEFAULTS = {
    "slouch_threshold": 0.15,
    "lean_threshold": 0.15,
    "shoulder_tilt_threshold": 0.10,
    "shoulder_width_threshold": 0.8,
    "auto_tune_lookback_days": 7,
    "auto_tune_target_events_per_day": 12.0,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-tune posture thresholds from logs.")
    parser.add_argument("--config", default="config.yaml", help="Path to app config JSON file")
    parser.add_argument("--logs-dir", default="logs", help="Directory with daily JSONL logs")
    parser.add_argument("--lookback-days", type=int, help="Override lookback days")
    parser.add_argument("--target-events", type=float, help="Override target events/day")
    parser.add_argument("--dry-run", action="store_true", help="Do not write config")
    parser.add_argument("--print", dest="print_report", action="store_true", help="Print report JSON")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    data = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    merged = dict(DEFAULTS)
    merged.update(data)
    return merged


def main():
    args = parse_args()
    config_path = Path(args.config)
    logs_dir = Path(args.logs_dir)
    raw = load_config(config_path)
    cfg_obj = SimpleNamespace(**raw)

    lookback_days = args.lookback_days or int(raw.get("auto_tune_lookback_days", DEFAULTS["auto_tune_lookback_days"]))
    target_events = args.target_events or float(
        raw.get("auto_tune_target_events_per_day", DEFAULTS["auto_tune_target_events_per_day"])
    )

    report = tune_thresholds(
        config_obj=cfg_obj,
        logs_dir=logs_dir,
        lookback_days=lookback_days,
        target_events_per_day=target_events,
    )

    if args.print_report:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(f"Auto-tune result: {report.get('reason', 'applied') if not report.get('applied') else 'applied'}")

    if report.get("applied") and not args.dry_run:
        updated = dict(raw)
        for key in DEFAULTS.keys():
            if hasattr(cfg_obj, key):
                updated[key] = getattr(cfg_obj, key)
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(updated, f, ensure_ascii=False)
        logs_dir.mkdir(parents=True, exist_ok=True)
        out = logs_dir / f"{Path(config_path).stem}.autotune.last.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"Config updated: {config_path}")
        print(f"Report saved: {out}")
        return 0

    return 0 if report.get("applied") else 1


if __name__ == "__main__":
    raise SystemExit(main())
