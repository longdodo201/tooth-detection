"""Run validation metrics and fixed-threshold diagnostic error counts for one full run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
from pathlib import Path
from typing import Any

from yolo_eval_common import (
    detection_error_counts,
    label_path_for_image,
    load_fdi_names,
    read_yolo_boxes,
    require_completed_run,
    resolve_run,
    run_name,
    sha256_file,
    validation_image_refs,
)

NMS_IOU_THRESHOLD = 0.70


def _sequence(value: Any) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value) if value is not None else []


def _number(value: Any) -> float | str:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return ""
    return result if math.isfinite(result) else ""


def per_class_rows(box_metrics: Any, class_names: list[str]) -> list[dict[str, Any]]:
    """Map installed Ultralytics DetMetrics.box per-class arrays to FDI IDs."""
    indices = [int(value) for value in _sequence(box_metrics.ap_class_index)]
    precision = _sequence(box_metrics.p)
    recall = _sequence(box_metrics.r)
    ap50 = _sequence(box_metrics.ap50)
    ap50_95 = _sequence(box_metrics.ap)
    observed = {class_id: index for index, class_id in enumerate(indices)}
    rows: list[dict[str, Any]] = []
    for class_id, class_name in enumerate(class_names):
        if class_id not in observed:
            rows.append({
                "class_id": class_id, "class_name": class_name, "class_status": "not_observed",
                "precision": "", "recall": "", "ap50": "", "ap50_95": "",
            })
            continue
        index = observed[class_id]
        rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "class_status": "available",
            "precision": _number(precision[index]) if index < len(precision) else "",
            "recall": _number(recall[index]) if index < len(recall) else "",
            "ap50": _number(ap50[index]) if index < len(ap50) else "",
            "ap50_95": _number(ap50_95[index]) if index < len(ap50_95) else "",
        })
    return rows


def _load_ground_truth(label_file: Path, width: int, height: int) -> tuple[list[int], list[tuple[float, float, float, float]]]:
    return read_yolo_boxes(label_file, width, height)


def _write_csv_exclusive(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(
    model_name: str,
    seed: int,
    run_root: Path,
    split_root: Path,
    mapping_path: Path,
    output_root: Path,
    confidence_threshold: float,
    match_iou_threshold: float,
) -> Path:
    """Validate one selected best checkpoint and write metrics plus diagnostic counts."""
    if not math.isfinite(confidence_threshold) or not 0 <= confidence_threshold <= 1:
        raise ValueError("error confidence threshold must be finite and in [0, 1]")
    if not math.isfinite(match_iou_threshold) or not 0 < match_iou_threshold <= 1:
        raise ValueError("error matching IoU threshold must be finite and in (0, 1]")
    run_dir = resolve_run(run_root, model_name, seed)
    checkpoint = require_completed_run(run_dir, model_name, seed)
    output_dir = output_root / run_name(model_name, seed)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation output: {output_dir}")

    class_names = load_fdi_names(mapping_path)
    image_refs = validation_image_refs(split_root)
    label_files = [label_path_for_image(split_root, image_ref) for image_ref, _ in image_refs]

    # Heavy imports are delayed so pure helpers remain testable without model execution.
    import cv2
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    model = YOLO(str(checkpoint.resolve()))
    metrics = model.val(
        data=str((split_root / "dataset.yaml").resolve()),
        split="val", imgsz=640, batch=1, device=0,
        conf=0.001, iou=NMS_IOU_THRESHOLD, max_det=300, plots=True,
        project=str(output_root.resolve()), name=run_name(model_name, seed),
        exist_ok=False, save_json=False, verbose=True,
    )
    actual_output_dir = output_dir

    box_metrics = metrics.box
    class_rows = per_class_rows(box_metrics, class_names)
    _write_csv_exclusive(
        actual_output_dir / "per_class_metrics.csv",
        ["class_id", "class_name", "class_status", "precision", "recall", "ap50", "ap50_95"],
        class_rows,
    )
    summary = {
        "model": model_name,
        "seed": seed,
        "run_directory": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "validation_manifest": str((split_root / "val.txt").resolve()),
        "validation_image_count": len(image_refs),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "python_version": platform.python_version(),
        "metrics": {
            "precision": _number(box_metrics.mp),
            "recall": _number(box_metrics.mr),
            "map50": _number(box_metrics.map50),
            "map50_95": _number(box_metrics.map),
        },
        "validation_settings": {
            "split": "val", "imgsz": 640, "batch": 1, "device": 0,
            "precision": "FP32", "confidence_floor": 0.001,
            "nms_iou": NMS_IOU_THRESHOLD, "max_det": 300,
        },
        "per_class_metrics_file": "per_class_metrics.csv",
        "ultralytics_metrics_are_standard": True,
        "custom_error_analysis_file": "detection_error_analysis.csv",
    }
    with (actual_output_dir / "validation_summary.json").open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")

    counts: dict[tuple[str, str, str], int] = {}
    for (_, image_path), label_file in zip(image_refs, label_files):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode validation image: {image_path}")
        height, width = image.shape[:2]
        true_classes, true_boxes = _load_ground_truth(label_file, width, height)
        prediction = model.predict(
            source=[image], imgsz=640, batch=1, device=0, quantize=32,
            conf=confidence_threshold, iou=NMS_IOU_THRESHOLD,
            save=False, verbose=False,
        )[0]
        if prediction.boxes is None:
            predicted_classes, predicted_boxes = [], []
        else:
            predicted_classes = [int(value) for value in prediction.boxes.cls.cpu().tolist()]
            predicted_boxes = [tuple(float(v) for v in box) for box in prediction.boxes.xyxy.cpu().tolist()]
        if any(not 0 <= class_id < len(class_names) for class_id in predicted_classes):
            raise ValueError(f"Prediction contains class ID outside the configured FDI mapping: {image_path}")
        image_counts = detection_error_counts(
            true_classes, true_boxes, predicted_classes, predicted_boxes, match_iou_threshold
        )
        for key, count in image_counts.items():
            counts[key] = counts.get(key, 0) + count

    error_rows = [
        {
            "error_type": kind,
            "true_class": class_names[int(true_class)] if true_class else "",
            "predicted_class": class_names[int(pred_class)] if pred_class else "",
            "count": count,
        }
        for (kind, true_class, pred_class), count in sorted(counts.items())
    ]
    _write_csv_exclusive(
        actual_output_dir / "detection_error_analysis.csv",
        ["error_type", "true_class", "predicted_class", "count"],
        error_rows,
    )
    error_metadata = {
        "custom_diagnostic_not_mAP": True,
        "matching_rule": "class-agnostic greedy one-to-one matching in descending box IoU; ties by validation label order then prediction order",
        "match_iou_threshold": match_iou_threshold,
        "confidence_threshold": confidence_threshold,
        "nms_iou_threshold": NMS_IOU_THRESHOLD,
        "error_types": ["correct_match", "class_confusion", "false_negative", "duplicate_prediction", "localization_error", "background_false_positive"],
        "counts_file": "detection_error_analysis.csv",
        "validation_images": len(image_refs),
        "thresholds_are_custom_diagnostic_choices_not_standard_mAP": True,
    }
    with (actual_output_dir / "detection_error_analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(error_metadata, stream, indent=2)
        stream.write("\n")
    return actual_output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("yolo11s", "yolo26s"), required=True)
    parser.add_argument("--seed", choices=(13, 21, 42), type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/full"))
    parser.add_argument("--mapping", type=Path, default=Path("generated/quadrant_enumeration_fdi_labels/class_mapping.json"))
    parser.add_argument("--output-root", type=Path, default=Path("runs/evaluation"))
    parser.add_argument("--error-confidence", type=float, default=0.25, help="Custom error-count confidence threshold (default: 0.25).")
    parser.add_argument("--error-match-iou", type=float, default=0.50, help="Class-agnostic box matching IoU in (0, 1] (default: 0.50).")
    args = parser.parse_args()
    try:
        output_dir = run_validation(
            args.model, args.seed, args.run_root, Path("generated/quadrant_enumeration_fdi_split"), args.mapping,
            args.output_root, args.error_confidence, args.error_match_iou,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Validation artifacts: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
