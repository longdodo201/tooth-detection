"""Measure in-memory single-image FP32 latency for one full-run best checkpoint."""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from yolo_eval_common import require_completed_run, resolve_run, sha256_file, validation_image_refs

IMAGE_SIZE = 640
BATCH_SIZE = 1
WARMUP_IMAGES = 20
TIMED_PASSES = 3


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    if not values:
        raise ValueError("values must not be empty")
    if not 0 < percentile <= 100:
        raise ValueError("percentile must be in (0, 100]")
    ordered = sorted(values)
    rank = math.ceil(percentile / 100 * len(ordered))
    return ordered[max(0, rank - 1)]


def _nvidia_smi_driver() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            check=False, capture_output=True, text=True, timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip().splitlines()[0] if result.returncode == 0 and result.stdout.strip() else None


def measure_latency(model_name: str, seed: int, run_root: Path, split_root: Path, output: Path) -> Path:
    run_dir = resolve_run(run_root, model_name, seed)
    checkpoint = require_completed_run(run_dir, model_name, seed)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite latency output: {output}")
    refs = validation_image_refs(split_root)

    import cv2
    import torch
    import torchvision
    import ultralytics
    from ultralytics import YOLO

    # Decode all validation images before model loading/warm-up/timing.
    images = []
    for _, image_path in refs:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode validation image: {image_path}")
        images.append(image)
    if len(images) != 127:
        raise ValueError(f"Expected 127 validation images, found {len(images)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; the approved latency procedure requires the RTX 2050")
    gpu = torch.cuda.get_device_properties(0)
    if "rtx 2050" not in gpu.name.lower():
        raise RuntimeError(f"Expected the approved RTX 2050 device 0, found {gpu.name!r}")

    model = YOLO(str(checkpoint.resolve()))

    def predict(image: Any) -> Any:
        return model.predict(
            source=[image], imgsz=IMAGE_SIZE, batch=BATCH_SIZE,
            device=0, quantize=32, conf=0.001, iou=0.7,
            save=False, verbose=False,
        )[0]

    for image in images[:WARMUP_IMAGES]:
        predict(image)
    torch.cuda.synchronize(0)

    latencies_ms: list[float] = []
    component_ms: dict[str, list[float]] = {"preprocess": [], "inference": [], "postprocess": []}
    for _ in range(TIMED_PASSES):
        for image in images:
            torch.cuda.synchronize(0)
            started = time.perf_counter()
            result = predict(image)
            torch.cuda.synchronize(0)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            for key in component_ms:
                value = result.speed.get(key) if result.speed else None
                if value is not None and math.isfinite(float(value)):
                    component_ms[key].append(float(value))

    driver_version = _nvidia_smi_driver()
    sorted_latency = sorted(latencies_ms)
    report = {
        "model": model_name,
        "seed": seed,
        "run_directory": str(run_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "validation_manifest": str((split_root / "val.txt").resolve()),
        "validation_image_count": len(images),
        "warmup_image_count": WARMUP_IMAGES,
        "timed_passes": TIMED_PASSES,
        "timed_predictions": len(latencies_ms),
        "image_size": IMAGE_SIZE,
        "batch_size": BATCH_SIZE,
        "precision": "FP32",
        "device": 0,
        "timing_scope": "per-image in-memory Ultralytics predict call; includes preprocessing, model execution, postprocessing, and Python prediction-call overhead; excludes disk decode/loading",
        "cuda_synchronized_before_and_after_each_timed_call": True,
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "latency_ms": {
            "mean": statistics.mean(latencies_ms),
            "median": statistics.median(latencies_ms),
            "p95_nearest_rank": percentile_nearest_rank(latencies_ms, 95),
        },
        "effective_images_per_second": 1000.0 / statistics.mean(latencies_ms),
        "ultralytics_reported_component_mean_ms": {
            key: statistics.mean(values) if values else None for key, values in component_ms.items()
        },
        "software": {
            "python": platform.python_version(),
            "os": platform.platform(),
            "ultralytics": ultralytics.__version__,
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "cuda_build": torch.version.cuda,
        },
        "hardware": {
            "gpu": gpu.name,
            "gpu_total_memory_bytes": gpu.total_memory,
            "driver_version": driver_version or "unavailable",
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("yolo11s", "yolo26s"), required=True)
    parser.add_argument("--seed", choices=(13, 21, 42), type=int, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("runs/full"))
    parser.add_argument("--output", type=Path, help="Defaults to latency.json inside the named run directory; existing files are refused.")
    args = parser.parse_args()
    run_dir = resolve_run(args.run_root, args.model, args.seed)
    output = args.output or run_dir / "latency.json"
    try:
        result = measure_latency(
            args.model, args.seed, args.run_root,
            Path("generated/quadrant_enumeration_fdi_split"), output,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Latency report: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
