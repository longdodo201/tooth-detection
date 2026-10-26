"""Shared, validation-only helpers for YOLO result and latency tools."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path, PurePosixPath

EXPECTED_FDI_CODES = [
    10 * quadrant + tooth
    for quadrant in range(1, 5)
    for tooth in range(1, 9)
]
EXPECTED_RUNS = [(model, seed) for model in ("yolo11s", "yolo26s") for seed in (13, 21, 42)]
EXPECTED_VALIDATION_IMAGES = 127


def load_fdi_names(mapping_path: Path) -> list[str]:
    """Load and validate the project's contiguous 32-class FDI mapping."""
    with mapping_path.open("r", encoding="utf-8") as stream:
        mapping = json.load(stream)
    entries = mapping.get("class_to_fdi")
    if not isinstance(entries, list) or len(entries) != len(EXPECTED_FDI_CODES):
        raise ValueError("class_mapping.json must contain 32 class_to_fdi entries")
    names: list[str] = []
    for expected_id, (entry, expected_code) in enumerate(zip(entries, EXPECTED_FDI_CODES)):
        if entry.get("class_id") != expected_id or entry.get("fdi_code") != expected_code:
            raise ValueError("class_mapping.json must use contiguous IDs ordered by the 32 FDI codes")
        names.append(f"FDI_{expected_code}")
    return names


def run_name(model: str, seed: int) -> str:
    if (model, seed) not in EXPECTED_RUNS:
        raise ValueError("Expected model/seed pair is yolo11s or yolo26s and seed 13, 21, or 42")
    return f"{model}_640_seed{seed}"


def resolve_run(run_root: Path, model: str, seed: int) -> Path:
    """Resolve only a named full-experiment run; pilot directories are never searched."""
    if run_root.name.lower() != "full":
        raise ValueError("run root must be a directory named 'full'; pilot results are excluded")
    return run_root / run_name(model, seed)


def require_completed_run(run_dir: Path, model: str, seed: int) -> Path:
    """Require the completion marker written by a normal return from the training command."""
    checkpoint = run_dir / "weights" / "best.pt"
    marker = run_dir / "run_summary.json"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Full-run best checkpoint not found: {checkpoint}")
    if not marker.is_file():
        raise FileNotFoundError(f"Completion marker not found: {marker}")
    with marker.open("r", encoding="utf-8") as stream:
        summary = json.load(stream)
    if not isinstance(summary, dict):
        raise ValueError(f"Run completion marker is not a JSON object: {marker}")
    if summary.get("status") != "completed" or summary.get("model") != model or summary.get("seed") != seed:
        raise ValueError(f"Run completion marker does not match {model} seed {seed}: {marker}")
    return checkpoint


def validation_image_refs(split_root: Path, expected_count: int = EXPECTED_VALIDATION_IMAGES) -> list[tuple[str, Path]]:
    """Read only val.txt and ensure every listed entry is under the image junction."""
    manifest = split_root / "val.txt"
    if not manifest.is_file():
        raise FileNotFoundError(f"Validation manifest not found: {manifest}")
    refs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for raw_line in manifest.read_text(encoding="utf-8").splitlines():
        value = raw_line.strip()
        if not value:
            continue
        relative = PurePosixPath(value.replace("\\", "/"))
        parts = [part for part in relative.parts if part not in (".", "")]
        if relative.is_absolute() or ".." in parts or not parts or parts[0] != "images":
            raise ValueError(f"Validation path must be relative to images/: {value}")
        normalized = Path(*parts).as_posix()
        if normalized in seen:
            raise ValueError(f"Duplicate validation image path: {value}")
        seen.add(normalized)
        image_path = split_root.joinpath(*parts)
        if not image_path.is_file():
            raise FileNotFoundError(f"Validation image is missing: {image_path}")
        refs.append((normalized, image_path))
    if len(refs) != expected_count:
        raise ValueError(f"Expected {expected_count} validation images, found {len(refs)}")
    return refs


def label_path_for_image(split_root: Path, image_ref: str) -> Path:
    relative = PurePosixPath(image_ref.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "images":
        raise ValueError(f"Image path is not a safe images/ path: {image_ref}")
    label_parts = list(relative.parts)
    label_parts[0] = "labels"
    label_parts[-1] = Path(label_parts[-1]).with_suffix(".txt").name
    result = split_root.joinpath(*label_parts)
    if not result.is_file():
        raise FileNotFoundError(f"Validation label is missing: {result}")
    return result


def read_yolo_boxes(label_file: Path, image_width: int, image_height: int) -> tuple[list[int], list[tuple[float, float, float, float]]]:
    """Read normalized YOLO xywh labels and convert them to pixel xyxy boxes."""
    classes: list[int] = []
    boxes: list[tuple[float, float, float, float]] = []
    for line_number, line in enumerate(label_file.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_file}:{line_number}: expected five YOLO label fields")
        try:
            class_id = int(fields[0])
            cx, cy, width, height = (float(value) for value in fields[1:])
        except ValueError as exc:
            raise ValueError(f"{label_file}:{line_number}: invalid numeric label") from exc
        if not 0 <= class_id < 32:
            raise ValueError(f"{label_file}:{line_number}: class ID must be 0..31")
        if not all(math.isfinite(value) for value in (cx, cy, width, height)):
            raise ValueError(f"{label_file}:{line_number}: coordinates must be finite")
        if width <= 0 or height <= 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and width <= 1 and height <= 1):
            raise ValueError(f"{label_file}:{line_number}: normalized box values are invalid")
        x1, y1 = (cx - width / 2) * image_width, (cy - height / 2) * image_height
        x2, y2 = (cx + width / 2) * image_width, (cy + height / 2) * image_height
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            raise ValueError(f"{label_file}:{line_number}: box is outside the image")
        classes.append(class_id)
        boxes.append((x1, y1, x2, y2))
    return classes, boxes


def box_iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_first = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    area_second = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = area_first + area_second - intersection
    return intersection / union if union > 0 else 0.0


def detection_error_counts(
    true_classes: list[int],
    true_boxes: list[tuple[float, float, float, float]],
    predicted_classes: list[int],
    predicted_boxes: list[tuple[float, float, float, float]],
    match_iou: float = 0.5,
) -> dict[tuple[str, str, str], int]:
    """Greedily match class-agnostic boxes by descending IoU, then count error types."""
    if len(true_classes) != len(true_boxes) or len(predicted_classes) != len(predicted_boxes):
        raise ValueError("Each class list must have one corresponding box per item")
    if not 0 < match_iou <= 1:
        raise ValueError("match_iou must be in (0, 1]")
    candidates = sorted(
        (
            (box_iou(gt_box, pred_box), gt_index, pred_index)
            for gt_index, gt_box in enumerate(true_boxes)
            for pred_index, pred_box in enumerate(predicted_boxes)
        ),
        key=lambda candidate: (-candidate[0], candidate[1], candidate[2]),
    )
    matched_gt: set[int] = set()
    matched_pred: set[int] = set()
    counts: dict[tuple[str, str, str], int] = {}

    def add(kind: str, true_class: str, pred_class: str) -> None:
        key = (kind, true_class, pred_class)
        counts[key] = counts.get(key, 0) + 1

    for overlap, gt_index, pred_index in candidates:
        if overlap < match_iou:
            break
        if gt_index in matched_gt or pred_index in matched_pred:
            continue
        matched_gt.add(gt_index)
        matched_pred.add(pred_index)
        gt_class, pred_class = true_classes[gt_index], predicted_classes[pred_index]
        add("correct_match" if gt_class == pred_class else "class_confusion", str(gt_class), str(pred_class))

    for pred_index, pred_box in enumerate(predicted_boxes):
        if pred_index in matched_pred:
            continue
        maximum_iou = max((box_iou(gt_box, pred_box) for gt_box in true_boxes), default=0.0)
        if maximum_iou >= match_iou:
            kind = "duplicate_prediction"
        elif maximum_iou > 0:
            kind = "localization_error"
        else:
            kind = "background_false_positive"
        add(kind, "", str(predicted_classes[pred_index]))

    for gt_index, gt_class in enumerate(true_classes):
        if gt_index not in matched_gt:
            add("false_negative", str(gt_class), "")
    return counts


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
