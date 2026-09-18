"""Convert Pascal VOC XML annotations to normalized YOLO labels."""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--classes", nargs=3, default=["geometry", "animal", "plant"])
    args = parser.parse_args()
    class_ids = {name: index for index, name in enumerate(args.classes)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for xml_path in sorted(args.xml_dir.glob("*.xml")):
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width, height = int(size.findtext("width")), int(size.findtext("height"))
        rows = []
        for obj in root.findall("object"):
            name = obj.findtext("name")
            if name not in class_ids:
                raise ValueError(f"Unknown class {name!r} in {xml_path}")
            box = obj.find("bndbox")
            xmin = max(0.0, float(box.findtext("xmin")))
            ymin = max(0.0, float(box.findtext("ymin")))
            xmax = min(float(width), float(box.findtext("xmax")))
            ymax = min(float(height), float(box.findtext("ymax")))
            if xmax <= xmin or ymax <= ymin:
                continue
            xc, yc = (xmin + xmax) / 2 / width, (ymin + ymax) / 2 / height
            rows.append(f"{class_ids[name]} {xc:.8f} {yc:.8f} {(xmax - xmin) / width:.8f} {(ymax - ymin) / height:.8f}")
        (args.output_dir / f"{xml_path.stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
