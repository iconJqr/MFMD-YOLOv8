"""Build a clean source ZIP after auditing its selected files."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_release import BAD_DIRS, BAD_SUFFIXES, audit


def eligible(relative: Path) -> bool:
    return not (set(relative.parts) & BAD_DIRS or relative.suffix.lower() in BAD_SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT.parent / "MFMD-YOLOv8-source.zip")
    args = parser.parse_args()
    issues = [item for item in audit(ROOT) if not item.startswith("generated/private directory:")]
    if issues:
        raise RuntimeError("Release audit failed:\n" + "\n".join(issues))
    files = sorted(path for path in ROOT.rglob("*") if path.is_file() and eligible(path.relative_to(ROOT)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path("MFMD-YOLOv8") / path.relative_to(ROOT))
    print(f"wrote {len(files)} files to {args.output}")


if __name__ == "__main__":
    main()
