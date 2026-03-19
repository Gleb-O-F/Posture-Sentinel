import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from environment_checks import build_environment_report  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Check local Posture Sentinel environment readiness.")
    parser.add_argument("--models-dir", default="models", help="Directory containing ONNX models")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index to probe when --check-camera is used")
    parser.add_argument("--check-camera", action="store_true", help="Attempt to open and read from the camera")
    parser.add_argument("--print", dest="print_report", action="store_true", help="Print JSON report")
    return parser.parse_args()


def main():
    args = parse_args()
    report = build_environment_report(
        models_dir=Path(args.models_dir),
        camera_id=args.camera_id,
        include_camera=args.check_camera,
    )

    if args.print_report or not report.get("ok"):
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("Environment OK")

    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
