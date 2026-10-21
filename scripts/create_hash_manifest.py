"""Create SHA-256 checksums for release files while excluding generated/private artifacts."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


EXCLUDED_PARTS = {
    ".git", ".DS_Store", "MFMD-YOLOv8", "__pycache__", ".pytest_cache", ".ipynb_checkpoints",
    "runs", "datasets", ".idea", ".vscode",
}
EXCLUDED_SUFFIXES = {".pt", ".onnx", ".engine", ".pyc"}


def included(path: Path) -> bool:
    return not (set(path.parts) & EXCLUDED_PARTS or path.suffix.lower() in EXCLUDED_SUFFIXES)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("SHA256SUMS.txt"))
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.resolve() != output.resolve()
                   and included(path.relative_to(root)))
    output.write_text("".join(f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in paths),
                      encoding="utf-8")
    print(f"wrote {len(paths)} hashes to {output}")


if __name__ == "__main__":
    main()
