import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reporting import build_daily_summary, build_range_summary, write_summary_file  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Posture Sentinel log summary.")
    parser.add_argument("--logs-dir", default="logs", help="Directory with daily JSONL logs (default: logs)")
    parser.add_argument("--date", help="Day in YYYY-MM-DD format")
    parser.add_argument("--from", dest="date_from", help="Start date in YYYY-MM-DD format")
    parser.add_argument("--to", dest="date_to", help="End date in YYYY-MM-DD format")
    parser.add_argument("--out", help="Output JSON path; default is logs/<label>.summary.json")
    parser.add_argument("--print", dest="print_summary", action="store_true", help="Print summary JSON to stdout")
    return parser.parse_args()


def main():
    args = parse_args()
    logs_dir = Path(args.logs_dir)

    summary = None
    label = None
    if args.date:
        summary = build_daily_summary(logs_dir, args.date)
        label = args.date
    elif args.date_from and args.date_to:
        summary = build_range_summary(logs_dir, args.date_from, args.date_to)
        label = f"{args.date_from}_to_{args.date_to}"
    else:
        print("Use either --date YYYY-MM-DD or --from YYYY-MM-DD --to YYYY-MM-DD")
        return 2

    if summary is None:
        print("No logs found for the requested period.")
        return 1

    if args.out:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    else:
        output_path = write_summary_file(logs_dir, f"{label}.summary.json", summary)

    print(f"Summary written: {output_path}")
    if args.print_summary:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
