"""Capture a machine-readable runtime description without collecting secrets."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def package_version(name: str):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("run_environment.json"))
    args = parser.parse_args()
    try:
        import torch
        gpu = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        torch_data = {"version": torch.__version__, "cuda": torch.version.cuda,
                      "cudnn": torch.backends.cudnn.version(), "gpus": gpu}
    except Exception as error:
        torch_data = {"error": type(error).__name__}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                  check=False, timeout=5).stdout.strip() or None
    except Exception:
        revision = None
    payload = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
        "executable_name": Path(sys.executable).name,
        "git_revision": revision,
        "torch": torch_data,
        "packages": {name: package_version(name) for name in
                     ("ultralytics", "torchvision", "numpy", "opencv-python", "pandas", "pyyaml")},
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
