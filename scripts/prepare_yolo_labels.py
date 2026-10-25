"""Validate and optionally write YOLO detection labels from COCO-like JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

from inspect_dataset import (
    bbox_xywh_to_normalized_xywh,
    inspect_dataset,
    png_dimensions,
)


def _category_number_map(
    definitions: list[dict[str, Any]], field_name: str, expected_values: range
) -> dict[int, tuple[int, Any]]:
    """Map source category ID to its parsed semantic number and original name."""
    by_id: dict[int, tuple[int, Any]] = {}
    numbers: list[int] = []
    for category in definitions:
        if "id" in category and "category_id" in category and category["id"] != category["category_id"]:
            raise ValueError(f"{field_name}: category id and category_id disagree: {category}")
        source_id = category.get("category_id", category.get("id"))
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            raise ValueError(f"{field_name}: category definition requires an integer id/category_id")
        if source_id in by_id:
            raise ValueError(f"{field_name}: duplicate category ID {source_id}")

        original_name = category.get("name")
        if isinstance(original_name, bool):
            raise ValueError(f"{field_name}: category name must parse as an integer, got {original_name!r}")
        if isinstance(original_name, int):
            number = original_name
        elif isinstance(original_name, str) and original_name.strip().isdigit():
            number = int(original_name.strip())
        else:
            raise ValueError(f"{field_name}: category name must parse as an integer, got {original_name!r}")
        if number not in expected_values:
            raise ValueError(
                f"{field_name}: category name {original_name!r} parses to {number}; "
                f"expected {min(expected_values)}–{max(expected_values)}"
            )
        by_id[source_id] = (number, original_name)
        numbers.append(number)

    if len(numbers) != len(set(numbers)) or set(numbers) != set(expected_values):
        raise ValueError(
            f"{field_name}: category names must map one-to-one to "
            f"{list(expected_values)}; observed parsed names {sorted(numbers)}"
        )
    return by_id


def make_fdi_class_mapping(data: dict[str, Any]) -> tuple[dict[tuple[int, int], int], dict[str, Any]]:
    """Map source category ID pairs to contiguous class IDs ordered by FDI code."""
    quadrants = _category_number_map(data["categories_1"], "categories_1", range(1, 5))
    tooth_numbers = _category_number_map(data["categories_2"], "categories_2", range(1, 9))

    entries = []
    for category_id_1, (quadrant, original_name_1) in quadrants.items():
        for category_id_2, (tooth_number, original_name_2) in tooth_numbers.items():
            entries.append(
                {
                    "fdi_code": 10 * quadrant + tooth_number,
                    "quadrant_number": quadrant,
                    "tooth_enumeration_number": tooth_number,
                    "original_category_id_1": category_id_1,
                    "original_category_id_2": category_id_2,
                    "category_name_1": original_name_1,
                    "category_name_2": original_name_2,
                }
            )
    entries.sort(key=lambda item: item["fdi_code"])
    for class_id, entry in enumerate(entries):
        entry["class_id"] = class_id

    pair_to_class = {
        (entry["original_category_id_1"], entry["original_category_id_2"]): entry["class_id"]
        for entry in entries
    }
    metadata = {
        "mode": "fdi_quadrant_enumeration",
        "fdi_formula": "fdi_code = 10 * quadrant_number + tooth_enumeration_number",
        "class_id_convention": "contiguous zero-based IDs ordered by ascending FDI code",
        "category_metadata_id_key": "category_id when present, otherwise id",
        "label_semantics": "category_id_1 is quadrant; category_id_2 is tooth enumeration, per the user-provided DENTEX description",
        "class_to_fdi": entries,
    }
    return pair_to_class, metadata


def observed_fdi_pairs(
    data: dict[str, Any], pair_to_class: dict[tuple[int, int], int]
) -> tuple[int, list[dict[str, int]]]:
    observed = {(item["category_id_1"], item["category_id_2"]) for item in data["annotations"]}
    unexpected = [
        {"category_id_1": pair[0], "category_id_2": pair[1]}
        for pair in sorted(observed)
        if pair not in pair_to_class
    ]
    return len(observed) - len(unexpected), unexpected


def build_label_texts(
    data: dict[str, Any],
    image_dir: Path,
    class_mapping: dict[tuple[int, int], int],
) -> dict[str, str]:
    """Create deterministic in-memory YOLO label text; writes no files."""
    image_by_id = {image["id"]: image for image in data["images"]}
    labels_by_name: dict[str, list[tuple[int, str]]] = {
        Path(image["file_name"]).with_suffix(".txt").name: [] for image in data["images"]
    }
    if len(labels_by_name) != len(data["images"]):
        raise ValueError("image filenames do not map to unique YOLO label filenames")

    dimensions_by_id: dict[int, tuple[int, int]] = {}
    for image in data["images"]:
        dimensions = png_dimensions(image_dir / image["file_name"])
        if dimensions != (image["width"], image["height"]):
            raise ValueError(f"image dimensions differ from JSON for {image['file_name']}")
        dimensions_by_id[image["id"]] = dimensions

    for annotation in sorted(data["annotations"], key=lambda item: item["id"]):
        image = image_by_id[annotation["image_id"]]
        width, height = dimensions_by_id[image["id"]]

        source_label = (annotation["category_id_1"], annotation["category_id_2"])
        if source_label not in class_mapping:
            raise ValueError(f"no class mapping for source label {source_label!r}")

        x_center, y_center, box_width, box_height = bbox_xywh_to_normalized_xywh(
            annotation["bbox"], width, height
        )
        x, y, pixel_width, pixel_height = annotation["bbox"]
        if x < 0 or y < 0 or x + pixel_width > width or y + pixel_height > height:
            raise ValueError(f"annotation {annotation['id']} bbox is outside image bounds")
        values = (x_center, y_center, box_width, box_height)
        if not all(0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"annotation {annotation['id']} normalizes outside [0, 1]")
        line = f"{class_mapping[source_label]} {x_center:.8f} {y_center:.8f} {box_width:.8f} {box_height:.8f}"
        label_name = Path(image["file_name"]).with_suffix(".txt").name
        labels_by_name[label_name].append((annotation["id"], line))

    return {
        name: "".join(line + "\n" for _, line in sorted(lines))
        for name, lines in sorted(labels_by_name.items())
    }


def write_label_bundle(output_dir: Path, labels: dict[str, str], class_metadata: dict[str, Any]) -> None:
    """Write labels and mapping metadata to a new directory, without overwriting."""
    if output_dir.exists():
        raise FileExistsError(f"output path already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".yolo-labels-", dir=output_dir.parent))
    try:
        labels_dir = staging / "labels"
        labels_dir.mkdir()
        for filename, content in labels.items():
            (labels_dir / filename).write_text(content, encoding="utf-8", newline="\n")
        (staging / "class_mapping.json").write_text(
            json.dumps(class_metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        (staging / "README.md").write_text(
            "# FDI YOLO bbox labels\n\n"
            "This directory contains label text only, generated from the selected annotation JSON. "
            "It does not contain source images, a train/validation split, or model configuration.\n\n"
            "Each file in `labels/` uses one YOLO detection row per object: "
            "`class_id x_center y_center width height`. Coordinates are normalized to the corresponding "
            "source image dimensions. `class_mapping.json` records the FDI code and original category IDs "
            "for every contiguous zero-based YOLO class ID. The mapping follows the user-provided DENTEX "
            "description and is not a disease classification.\n",
            encoding="utf-8",
            newline="\n",
        )
        os.replace(staging, output_dir)
    except Exception:
        for item in staging.rglob("*"):
            if item.is_file():
                item.unlink()
        for item in sorted(staging.rglob("*"), reverse=True):
            if item.is_dir():
                item.rmdir()
        staging.rmdir()
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--fdi", action="store_true", help="map category definitions to FDI codes")
    parser.add_argument("--dry-run", action="store_true", help="validate and report without writing")
    parser.add_argument("--output", type=Path, help="new output directory for labels and mapping metadata")
    args = parser.parse_args()

    if not args.dry_run and not args.fdi:
        parser.error("writing labels requires explicit --fdi mapping selection")
    if not args.dry_run and args.output is None:
        parser.error("--output is required when --dry-run is not used")
    if args.output is not None and not args.dry_run:
        repository_root = Path(__file__).resolve().parents[1]
        generated_root = (repository_root / "generated").resolve()
        output_resolved = args.output.resolve()
        try:
            output_resolved.relative_to(generated_root)
        except ValueError:
            parser.error(f"--output must be inside {generated_root}")
        if args.images.resolve() == output_resolved or args.images.resolve() in output_resolved.parents:
            parser.error("--output cannot be inside the source image directory")

    try:
        report, errors = inspect_dataset(args.annotations, args.images)
        if errors:
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
        with args.annotations.open("r", encoding="utf-8") as json_file:
            data = json.load(json_file)

        if args.fdi:
            class_mapping, class_metadata = make_fdi_class_mapping(data)
            valid_pair_count, unexpected_pairs = observed_fdi_pairs(data, class_mapping)
            class_metadata["observed_valid_fdi_pair_count"] = valid_pair_count
            class_metadata["unexpected_observed_pairs"] = unexpected_pairs
            if unexpected_pairs:
                print(
                    json.dumps(
                        {"valid_fdi_pair_count": valid_pair_count, "unexpected_pairs": unexpected_pairs},
                        ensure_ascii=False,
                        indent=2,
                    )
                )
                return 1
            labels = build_label_texts(data, args.images, class_mapping)
            line_count = sum(content.count("\n") for content in labels.values())
        else:
            class_metadata = None
            labels = None
            valid_pair_count = None
            unexpected_pairs = None
            line_count = report["annotation_count"]

        result = {
            "mode": "dry-run" if args.dry_run else "write",
            "image_count": report["image_count"],
            "annotation_count": report["annotation_count"],
            "label_files_would_generate": report["image_count"],
            "label_lines_would_generate": line_count,
            "class_mapping_selected": class_metadata is not None,
            "class_mapping": class_metadata,
            "valid_fdi_pair_count": valid_pair_count,
            "unexpected_pairs": unexpected_pairs,
            "output_written": False,
        }
        if args.dry_run:
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        assert labels is not None and class_metadata is not None
        write_label_bundle(args.output, labels, class_metadata)
        result["output_written"] = True
        result["output_dir"] = str(args.output.resolve())
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
