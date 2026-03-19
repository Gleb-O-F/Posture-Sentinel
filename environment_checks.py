import importlib.util
import sys
from pathlib import Path
from typing import Iterable

REQUIRED_MODULES = {
    "cv2": "opencv-python",
    "mediapipe": "mediapipe",
    "numpy": "numpy",
    "onnxruntime": "onnxruntime-directml/onnxruntime",
    "pystray": "pystray",
    "PIL": "Pillow",
}

WINDOWS_ONLY_MODULES = {
    "win32gui": "pywin32",
    "win32con": "pywin32",
}

REQUIRED_MODEL_FILES = [
    "pose_landmarks_detector_full.onnx",
]


def get_python_info() -> dict:
    return {
        "executable": sys.executable,
        "version": sys.version.split()[0],
        "version_info": {
            "major": sys.version_info.major,
            "minor": sys.version_info.minor,
            "micro": sys.version_info.micro,
        },
        "supported": (sys.version_info.major, sys.version_info.minor) >= (3, 10),
    }


def check_modules(platform: str = sys.platform) -> dict:
    modules = dict(REQUIRED_MODULES)
    if platform.startswith("win"):
        modules.update(WINDOWS_ONLY_MODULES)

    available = {}
    missing = {}
    for module_name, package_name in modules.items():
        if importlib.util.find_spec(module_name) is None:
            missing[module_name] = package_name
        else:
            available[module_name] = package_name

    return {
        "available": available,
        "missing": missing,
        "ok": not missing,
    }


def check_models(models_dir: Path, required_files: Iterable[str] = REQUIRED_MODEL_FILES) -> dict:
    present = []
    missing = []

    for filename in required_files:
        if (models_dir / filename).exists():
            present.append(filename)
        else:
            missing.append(filename)

    return {
        "models_dir": str(models_dir),
        "present": present,
        "missing": missing,
        "ok": not missing,
    }


def check_camera(camera_id: int = 0) -> dict:
    spec = importlib.util.find_spec("cv2")
    if spec is None:
        return {
            "checked": False,
            "ok": False,
            "reason": "cv2_unavailable",
        }

    import cv2  # type: ignore

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        cap.release()
        return {
            "checked": True,
            "ok": False,
            "camera_id": camera_id,
            "reason": "open_failed",
        }

    ret, _frame = cap.read()
    cap.release()
    return {
        "checked": True,
        "ok": bool(ret),
        "camera_id": camera_id,
        "reason": "read_ok" if ret else "read_failed",
    }


def build_environment_report(models_dir: Path, camera_id: int = 0, include_camera: bool = False, platform: str = sys.platform) -> dict:
    python_info = get_python_info()
    module_info = check_modules(platform=platform)
    model_info = check_models(models_dir)
    camera_info = check_camera(camera_id) if include_camera else {"checked": False, "ok": None}

    overall_ok = python_info["supported"] and module_info["ok"] and model_info["ok"]
    if include_camera:
        overall_ok = overall_ok and bool(camera_info["ok"])

    return {
        "ok": overall_ok,
        "python": python_info,
        "modules": module_info,
        "models": model_info,
        "camera": camera_info,
    }
