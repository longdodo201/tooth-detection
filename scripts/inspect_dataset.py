"""Read-only structural checks for the quadrant_enumeration COCO-like JSON."""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def bbox_xywh_to_normalized_xywh(
    bbox: list[float], image_width: int, image_height: int
) -> tuple[float, float, float, float]:
    """Convert pixel xywh to normalized YOLO center-x, center-y, width, height."""
    if len(bbox) != 4:
        raise ValueError("bbox must contain [x, y, width, height]")
    x, y, width, height = bbox
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    values = (x, y, width, height)
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("bbox values must be numbers")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("bbox values must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")
    return (
        (x + width / 2) / image_width,
        (y + height / 2) / image_height,
        width / image_width,
        height / image_height,
    )


def png_dimensions(path: Path) -> tuple[int, int]:
    """Read PNG width and height from its IHDR header without image libraries."""
    with path.open("rb") as image_file:
        header = image_file.read(24)
    if len(header) < 24 or header[:8] != PNG_SIGNATURE or header[12:16] != b"IHDR":
        raise ValueError(f"not a valid PNG header: {path}")
    return struct.unpack(">II", header[16:24])


def inspect_dataset(annotation_path: Path, image_dir: Path) -> tuple[dict[str, Any], list[str]]:
    """Inspect dataset structure and return a JSON-serializable report and errors."""
    with annotation_path.open("r", encoding="utf-8") as json_file:
        data = json.load(json_file)

    errors: list[str] = []
    for key in ("images", "annotations", "categories_1", "categories_2"):
        if not isinstance(data.get(key), list):
            errors.append(f"top-level field {key!r} must be a list")
    if errors:
        return {"errors": errors}, errors

    images = data["images"]
    annotations = data["annotations"]
    image_ids = [item.get("id") for item in images]
    annotation_ids = [item.get("id") for item in annotations]
    if len(set(image_ids)) != len(image_ids):
        errors.append("duplicate image IDs")
    if len(set(annotation_ids)) != len(annotation_ids):
        errors.append("duplicate annotation IDs")

    image_by_id = {item.get("id"): item for item in images}
    category_ids = {
        key: {item.get("category_id", item.get("id")) for item in data[key]}
        for key in ("categories_1", "categories_2")
    }
    if any(
        len(category_ids[key]) != len(data[key])
        or any(item.get("category_id", item.get("id")) is None for item in data[key])
        for key in category_ids
    ):
        errors.append("missing or duplicate category IDs")

    missing_images: list[str] = []
    dimension_mismatches: list[str] = []
    for image in images:
        filename = image.get("file_name")
        if not isinstance(filename, str) or Path(filename).name != filename:
            errors.append(f"invalid image file_name for image ID {image.get('id')!r}")
            continue
        path = image_dir / filename
        if not path.is_file():
            missing_images.append(filename)
            continue
        try:
            width, height = png_dimensions(path)
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
            continue
        if (width, height) != (image.get("width"), image.get("height")):
            dimension_mismatches.append(filename)

    for annotation in annotations:
        annotation_id = annotation.get("id")
        image = image_by_id.get(annotation.get("image_id"))
        if image is None:
            errors.append(f"annotation {annotation_id!r} references unknown image ID")
            continue
        for key in ("category_id_1", "category_id_2"):
            category_key = "categories_1" if key.endswith("_1") else "categories_2"
            if annotation.get(key) not in category_ids[category_key]:
                errors.append(f"annotation {annotation_id!r} has undefined {key}")
        try:
            bbox = annotation["bbox"]
            bbox_xywh_to_normalized_xywh(bbox, image["width"], image["height"])
            x, y, width, height = bbox
            if x < 0 or y < 0 or x + width > image["width"] or y + height > image["height"]:
                errors.append(f"annotation {annotation_id!r} bbox is outside image bounds")
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"annotation {annotation_id!r} has invalid bbox: {exc}")

    if missing_images:
        errors.append(f"{len(missing_images)} referenced image file(s) are missing")
    if dimension_mismatches:
        errors.append(f"{len(dimension_mismatches)} image dimension mismatch(es)")

    report = {
        "image_count": len(images),
        "annotation_count": len(annotations),
        "image_dimensions": {
            f"{width}x{height}": count
            for (width, height), count in Counter(
                (item.get("width"), item.get("height")) for item in images
            ).items()
        },
        "categories_1": data["categories_1"],
        "categories_2": data["categories_2"],
        "annotation_fields": sorted({key for item in annotations for key in item}),
        "missing_images": missing_images,
        "dimension_mismatches": dimension_mismatches,
        "errors": errors,
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path, help="source annotation JSON")
    parser.add_argument("--images", required=True, type=Path, help="directory containing source PNGs")
    args = parser.parse_args()
    try:
        report, errors = inspect_dataset(args.annotations, args.images)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
