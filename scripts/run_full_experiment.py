"""Run one or all six approved full YOLO experiments, sequentially."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS = ("yolo11s", "yolo26s")
SEEDS = (13, 21, 42)
DATASET = Path("generated/quadrant_enumeration_fdi_split/dataset.yaml")
CHECKPOINT_DIR = Path(".venv/models")

TRAIN_ARGUMENTS: dict[str, Any] = {
    "task": "detect", "pretrained": True, "cls_remap": True, "fraction": 1.0,
    "epochs": 50, "patience": 20, "imgsz": 640, "batch": 1, "nbs": 1,
    "deterministic": True, "device": 0, "workers": 0,
    "optimizer": "SGD", "lr0": 0.001, "lrf": 0.01, "cos_lr": False,
    "momentum": 0.937, "weight_decay": 0.0005,
    "warmup_epochs": 0.0, "warmup_momentum": 0.8, "warmup_bias_lr": 0.1,
    "amp": True, "cache": False, "val": True, "split": "val",
    "save": True, "save_period": -1, "plots": True, "resume": False,
    "exist_ok": True,
    "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.0,
    "degrees": 0.0, "translate": 0.0, "scale": 0.0, "shear": 0.0,
    "perspective": 0.0, "flipud": 0.0, "fliplr": 0.0, "bgr": 0.0,
    "mosaic": 0.0, "mixup": 0.0, "copy_paste": 0.0, "cutmix": 0.0,
    "auto_augment": None, "erasing": 0.0, "close_mosaic": 0,
}


class Tee:
    """Write a run log while keeping progress visible in the console."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_name(model: str, seed: int) -> str:
    if model not in MODELS or seed not in SEEDS:
        raise ValueError("Model/seed must be yolo11s or yolo26s and 13, 21, or 42")
    return f"{model}_640_seed{seed}"


def _summary_path(run_dir: Path) -> Path:
    return run_dir / "run_summary.json"


def _write_summary(path: Path, summary: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run_one(model_name: str, seed: int, runs_root: Path = Path("runs/full")) -> Path:
    """Run a single approved experiment; no output directory is ever reused."""
    name = run_name(model_name, seed)
    if Path.cwd().resolve() != REPO_ROOT:
        raise RuntimeError(f"Run this launcher from the repository root: {REPO_ROOT}")
    dataset = REPO_ROOT / DATASET
    checkpoint = REPO_ROOT / CHECKPOINT_DIR / f"{model_name}.pt"
    run_dir = (REPO_ROOT / runs_root / name).resolve()
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset configuration is missing: {dataset}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Local checkpoint is missing; no download attempted: {checkpoint}")
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")

    # mkdir is the exclusive reservation. Ultralytics may then write into only
    # this newly reserved directory; later invocations cannot reuse it.
    run_dir.mkdir(parents=True, exist_ok=False)
    log_path = run_dir / "training.log"
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    actual_arguments = dict(TRAIN_ARGUMENTS, data=str(dataset), project=str(run_dir.parent), name=name, seed=seed)
    metadata: dict[str, Any] = {
        "status": "running", "model": model_name, "seed": seed,
        "run_name": name, "started_at_utc": started_at,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": sha256_file(checkpoint),
        "dataset_yaml": str(dataset), "train_arguments_passed": actual_arguments,
        "python_version": platform.python_version(), "platform": platform.platform(),
    }

    try:
        with log_path.open("x", encoding="utf-8", buffering=1) as log_stream:
            tee_out = Tee(sys.stdout, log_stream)
            tee_err = Tee(sys.stderr, log_stream)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                # Imports happen only after all path/refusal checks; tests replace
                # them with fakes so no model/GPU is initialized.
                import torch
                import torchvision
                import ultralytics
                from ultralytics import YOLO

                metadata["software"] = {
                    "ultralytics": ultralytics.__version__, "torch": torch.__version__,
                    "torchvision": torchvision.__version__, "cuda_build": torch.version.cuda,
                }
                metadata["hardware"] = {
                    "cuda_available": bool(torch.cuda.is_available()),
                    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                }
                print(f"Starting {model_name}, seed {seed}; output: {run_dir}")
                yolo = YOLO(str(checkpoint))
                training_started = time.perf_counter()
                yolo.train(**actual_arguments)
                metadata["training_duration_seconds"] = time.perf_counter() - training_started

        required = (
            run_dir / "args.yaml", run_dir / "results.csv",
            run_dir / "weights" / "best.pt", run_dir / "weights" / "last.pt",
        )
        missing = [str(path.relative_to(run_dir)) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError("Training returned but required artifacts are missing: " + ", ".join(missing))
        # Preserve Ultralytics' resolved argument file beside the completion marker.
        metadata["resolved_args_file"] = "args.yaml"
        metadata["artifacts"] = {
            "training_log": "training.log", "args": "args.yaml", "results": "results.csv",
            "best_checkpoint": "weights/best.pt", "last_checkpoint": "weights/last.pt",
        }
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "completed"
        _write_summary(_summary_path(run_dir), metadata)
        return run_dir
    except BaseException as exc:
        metadata["elapsed_seconds"] = time.perf_counter() - started
        metadata["failed_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata["status"] = "failed"
        metadata["error_type"] = type(exc).__name__
        message = str(exc)
        if "out of memory" in message.lower() or "cuda oom" in message.lower():
            metadata["failure_category"] = "cuda_oom"
            message += " (CUDA OOM: launcher stopped without changing batch size or settings)"
        metadata["error_message"] = message
        metadata["traceback"] = traceback.format_exc()
        try:
            _write_summary(_summary_path(run_dir), metadata)
        except FileExistsError:
            # A completion marker must never be overwritten.
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Run all six model/seed pairs sequentially.")
    selection.add_argument("--model", choices=MODELS, help="Run one selected model and seed.")
    parser.add_argument("--seed", type=int, choices=SEEDS, help="Required with --model.")
    args = parser.parse_args(argv)
    if args.model and args.seed is None:
        parser.error("--seed is required with --model")
    if args.all and args.seed is not None:
        parser.error("--seed can only be used with --model")
    pairs = [(model, seed) for model in MODELS for seed in SEEDS] if args.all else [(args.model, args.seed)]
    for model, seed in pairs:
        try:
            path = run_one(model, seed)
        except BaseException as exc:
            print(f"FAILED {model} seed {seed}: {exc}", file=sys.stderr)
            return 1
        print(f"COMPLETED {model} seed {seed}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
