"""Validate FDI YOLO labels and create reproducible image-level split manifests."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_FDI_CODES = [10 * quadrant + tooth for quadrant in range(1, 5) for tooth in range(1, 9)]


def load_fdi_mapping(path: Path) -> list[dict[str, Any]]:
    """Validate that the mapping is the expected ordered 32-class FDI map."""
    with path.open("r", encoding="utf-8") as mapping_file:
        metadata = json.load(mapping_file)
    entries = metadata.get("class_to_fdi")
    if not isinstance(entries, list):
        raise ValueError("class_mapping.json must contain a class_to_fdi list")
    if [entry.get("class_id") for entry in entries] != list(range(32)):
        raise ValueError("FDI class IDs must be contiguous from 0 through 31")
    if [entry.get("fdi_code") for entry in entries] != EXPECTED_FDI_CODES:
        raise ValueError("mapping must contain the 32 expected FDI codes in ascending order")
    for entry in entries:
        quadrant = entry.get("quadrant_number")
        tooth = entry.get("tooth_enumeration_number")
        if (
            isinstance(quadrant, bool)
            or not isinstance(quadrant, int)
            or not 1 <= quadrant <= 4
            or isinstance(tooth, bool)
            or not isinstance(tooth, int)
            or not 1 <= tooth <= 8
        ):
            raise ValueError(f"invalid quadrant/tooth enumeration in mapping entry {entry!r}")
        if entry.get("fdi_code") != 10 * quadrant + tooth:
            raise ValueError(f"FDI formula mismatch in mapping entry {entry!r}")
        for key in ("original_category_id_1", "original_category_id_2"):
            value = entry.get(key)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"mapping entry requires integer {key}")
    return entries


def validate_images_and_labels(
    images_dir: Path, labels_dir: Path, mapping_entries: list[dict[str, Any]]
) -> tuple[list[str], Counter[int], dict[str, int]]:
    """Check paired files and every YOLO row; blank label files are valid negatives."""
    images = sorted(path.name for path in images_dir.glob("*.png") if path.is_file())
    if not images:
        raise ValueError(f"no PNG images found in {images_dir}")
    image_stems = {Path(name).stem for name in images}
    if len(image_stems) != len(images):
        raise ValueError("image basenames are not unique")

    labels = {path.name for path in labels_dir.glob("*.txt") if path.is_file()}
    expected_labels = {f"{Path(name).stem}.txt" for name in images}
    missing = sorted(expected_labels - labels)
    extra = sorted(labels - expected_labels)
    if missing or extra:
        raise ValueError(f"image/label mismatch: {len(missing)} missing labels, {len(extra)} extra labels")

    class_counts: Counter[int] = Counter()
    rows_by_image: dict[str, int] = {}
    for image_name in images:
        label_path = labels_dir / f"{Path(image_name).stem}.txt"
        row_count = 0
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"{label_path.name}:{line_number}: expected 5 columns")
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"{label_path.name}:{line_number}: non-numeric label value") from exc
            if not 0 <= class_id < 32:
                raise ValueError(f"{label_path.name}:{line_number}: class ID outside 0–31")
            if not all(math.isfinite(value) for value in coordinates):
                raise ValueError(f"{label_path.name}:{line_number}: non-finite coordinate")
            center_x, center_y, width, height = coordinates
            if not all(0.0 <= value <= 1.0 for value in coordinates) or width <= 0 or height <= 0:
                raise ValueError(f"{label_path.name}:{line_number}: invalid normalized bbox values")
            if (
                center_x - width / 2 < -1e-6
                or center_y - height / 2 < -1e-6
                or center_x + width / 2 > 1 + 1e-6
                or center_y + height / 2 > 1 + 1e-6
            ):
                raise ValueError(f"{label_path.name}:{line_number}: normalized bbox extends outside image")
            class_counts[class_id] += 1
            row_count += 1
        rows_by_image[image_name] = row_count
    if sum(class_counts.values()) == 0:
        raise ValueError("no annotation rows found in label files")
    return images, class_counts, rows_by_image


def split_image_names(
    image_names: list[str], seed: int = 42, train_fraction: float = 0.8
) -> tuple[list[str], list[str]]:
    """Deterministically shuffle names, floor the train count, then sort each split."""
    if len(set(image_names)) != len(image_names):
        raise ValueError("image names must be unique")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    shuffled = sorted(image_names)
    random.Random(seed).shuffle(shuffled)
    train_count = int(len(shuffled) * train_fraction)
    return sorted(shuffled[:train_count]), sorted(shuffled[train_count:])


def summarize_split(
    names: list[str], rows_by_image: dict[str, int], labels_dir: Path
) -> tuple[int, Counter[int]]:
    annotation_count = 0
    classes: Counter[int] = Counter()
    for name in names:
        annotation_count += rows_by_image[name]
        label_path = labels_dir / f"{Path(name).stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                classes[int(line.split()[0])] += 1
    return annotation_count, classes


def write_split_artifacts(
    output_dir: Path,
    train_names: list[str],
    val_names: list[str],
    mapping_entries: list[dict[str, Any]],
    train_annotations: int,
    val_annotations: int,
    train_classes: Counter[int],
    val_classes: Counter[int],
    seed: int,
    train_fraction: float,
    images_dir: Path,
    labels_dir: Path,
    mapping_path: Path,
) -> None:
    """Write split files to a new directory. Refuse to overwrite any existing path."""
    if output_dir.exists():
        raise FileExistsError(f"split output already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    repository_root = Path(__file__).resolve().parents[1]
    try:
        dataset_root = os.path.relpath(output_dir.resolve(), start=repository_root).replace("\\", "/")
    except ValueError:  # output is on a different Windows drive
        dataset_root = output_dir.resolve().as_posix()

    (output_dir / "train.txt").write_text(
        "".join(f"./images/{name}\n" for name in train_names), encoding="utf-8", newline="\n"
    )
    (output_dir / "val.txt").write_text(
        "".join(f"./images/{name}\n" for name in val_names), encoding="utf-8", newline="\n"
    )
    (output_dir / "dataset.yaml").write_text(
        f"path: {dataset_root}\ntrain: train.txt\nval: val.txt\nnames:\n"
        + "".join(f"  {entry['class_id']}: FDI_{entry['fdi_code']}\n" for entry in mapping_entries),
        encoding="utf-8",
        newline="\n",
    )
    class_to_fdi = {entry["class_id"]: entry["fdi_code"] for entry in mapping_entries}
    summary = {
        "split_strategy": "deterministic image-level random split",
        "grouping_metadata_found": False,
        "group_independence_guaranteed": False,
        "limitation": "No patient/study/group identifier was found; image-level split cannot rule out patient-level leakage.",
        "seed": seed,
        "train_fraction_requested": train_fraction,
        "train_rounding": "floor(total_images * train_fraction)",
        "image_count": {"train": len(train_names), "val": len(val_names)},
        "annotation_count": {"train": train_annotations, "val": val_annotations},
        "class_distribution_annotations": {
            split_name: {
                str(class_id): {"fdi_code": class_to_fdi[class_id], "count": class_counter[class_id]}
                for class_id in sorted(class_to_fdi)
            }
            for split_name, class_counter in (("train", train_classes), ("val", val_classes))
        },
        "source_image_dir": str(images_dir.resolve()),
        "source_label_dir": str(labels_dir.resolve()),
        "class_mapping_file": str(mapping_path.resolve()),
        "test_dataset_used": False,
        "empty_label_behavior": "A zero-byte label file is accepted as an image with no objects; it contributes zero annotations.",
    }
    (output_dir / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (output_dir / ".gitignore").write_text("/images/\n/labels/\n", encoding="utf-8", newline="\n")
    readme = f"""# Quadrant enumeration FDI train/validation split

This directory contains reproducible split manifests and a YOLO dataset YAML. It does not copy image or label files.

- Strategy: deterministic image-level 80/20 split; seed `{seed}`.
- Train/validation images: {len(train_names)} / {len(val_names)}.
- Train/validation annotations: {train_annotations} / {val_annotations}.
- Group metadata: none was found; patient-level independence cannot be guaranteed.
- Test data was not used.
- Blank label files are accepted and count as zero annotations; this source currently has no blank label files.

The `dataset.yaml` path is relative to the repository root, and its train/validation entries are relative to that dataset root. Run Ultralytics commands from the repository root; running from this split directory would interpret the path differently. The manifests and junctions use the standard YOLO layout. Create local NTFS junctions before using them so `images/` points to the source PNG directory and `labels/` points to the FDI label bundle. The junction paths are ignored by Git and can be recreated from `split_summary.json`:

```powershell
New-Item -ItemType Junction -Path .\\images -Target '{images_dir.resolve()}'
New-Item -ItemType Junction -Path .\\labels -Target '{labels_dir.resolve()}'
```

`train.txt` and `val.txt` list image paths relative to this directory. YOLO resolves labels by replacing the `images` path component with `labels`; no source image or label copies are required.
"""
    (output_dir / "README.md").write_text(readme, encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    args = parser.parse_args()

    try:
        if args.output.exists():
            raise FileExistsError(f"split output already exists: {args.output}")
        mapping_entries = load_fdi_mapping(args.mapping)
        image_names, total_classes, rows_by_image = validate_images_and_labels(
            args.images, args.labels, mapping_entries
        )
        train_names, val_names = split_image_names(image_names, args.seed, args.train_fraction)
        train_annotations, train_classes = summarize_split(train_names, rows_by_image, args.labels)
        val_annotations, val_classes = summarize_split(val_names, rows_by_image, args.labels)
        if set(train_names) & set(val_names) or len(train_names) + len(val_names) != len(image_names):
            raise ValueError("split is not a complete disjoint partition of source images")

        write_split_artifacts(
            args.output,
            train_names,
            val_names,
            mapping_entries,
            train_annotations,
            val_annotations,
            train_classes,
            val_classes,
            args.seed,
            args.train_fraction,
            args.images,
            args.labels,
            args.mapping,
        )
        result = {
            "output_dir": str(args.output.resolve()),
            "seed": args.seed,
            "image_count": {"train": len(train_names), "val": len(val_names)},
            "annotation_count": {"train": train_annotations, "val": val_annotations},
            "class_count": len(mapping_entries),
            "missing_files": 0,
            "invalid_labels": 0,
            "grouping_metadata_found": False,
            "test_dataset_used": False,
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
