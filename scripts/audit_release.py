"""Fail on common private/generated artifacts and obvious machine-specific paths."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


BAD_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", "runs", "datasets", ".idea", ".vscode"}
BAD_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".engine", ".pyc"}
TEXT_SUFFIXES = {".py", ".md", ".yaml", ".yml", ".txt", ".csv", ".toml", ".cfg", ".cff", ".in"}
PRIVATE_PATTERNS = (
    re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", re.IGNORECASE),
    re.compile(r"/Users/[^/\s]+"),
    re.compile(r"/home/[^/\s]+"),
)


def audit(root: Path) -> list[str]:
    issues = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.resolve() == Path(__file__).resolve():
            continue  # The scanner's own regular-expression literals resemble paths by design.
        if set(relative.parts) & BAD_DIRS:
            issues.append(f"generated/private directory: {relative}")
            continue
        if path.is_file() and path.suffix.lower() in BAD_SUFFIXES:
            issues.append(f"excluded artifact: {relative}")
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES:
            text = path.read_text(encoding="utf-8", errors="ignore")
            for pattern in PRIVATE_PATTERNS:
                if pattern.search(text):
                    issues.append(f"machine-specific path: {relative}")
                    break
    return sorted(set(issues))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--allow-generated", action="store_true", help="Ignore cache directories during local checks")
    args = parser.parse_args()
    issues = audit(args.root.resolve())
    if args.allow_generated:
        issues = [item for item in issues if not item.startswith("generated/private directory:")]
    if issues:
        print("\n".join(issues))
        raise SystemExit(1)
    print("release audit passed")


if __name__ == "__main__":
    main()
