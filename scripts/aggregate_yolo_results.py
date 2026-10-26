"""Aggregate only the six named full-experiment YOLO run directories."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from yolo_eval_common import EXPECTED_FDI_CODES, EXPECTED_RUNS, run_name, sha256_file

METRIC_COLUMNS = {
    "map50": "metrics/mAP50(B)",
    "map50_95": "metrics/mAP50-95(B)",
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
}
LATENCY_FIELDS = {
    "latency_mean_ms": "mean",
    "latency_median_ms": "median",
    "latency_p95_ms": "p95_nearest_rank",
    "latency_images_per_second": "images_per_second",
}
CLASS_METRICS = ("precision", "recall", "ap50", "ap50_95")


def finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def read_best_validation_row(results_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return the latest row attaining highest validation box mAP50-95."""
    try:
        with results_path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames:
                return None, "results.csv has no header"
            required = [METRIC_COLUMNS[name] for name in METRIC_COLUMNS]
            missing = [name for name in required if name not in reader.fieldnames]
            if missing:
                return None, "results.csv missing metric columns: " + ", ".join(missing)
            best: dict[str, Any] | None = None
            best_map = -math.inf
            best_epoch = -math.inf
            for row_index, row in enumerate(reader, start=1):
                score = finite_float(row.get(METRIC_COLUMNS["map50_95"]))
                if score is None:
                    continue
                epoch = finite_float(row.get("epoch"))
                epoch_key = epoch if epoch is not None else float(row_index)
                if score > best_map or (score == best_map and epoch_key >= best_epoch):
                    best, best_map, best_epoch = row, score, epoch_key
            return best, None if best is not None else "results.csv contains no finite mAP50-95 row"
    except OSError as exc:
        return None, f"cannot read results.csv: {exc}"


def _read_summary(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            summary = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"run_summary.json unavailable or invalid: {exc}"
    if not isinstance(summary, dict):
        return None, "run_summary.json must contain an object"
    return summary, None


def collect_run(run_dir: Path, model: str, seed: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "run_status": "missing_run",
        "status_detail": "run directory does not exist",
        "best_epoch": "",
        "training_duration_seconds": "",
        **{metric: "" for metric in METRIC_COLUMNS},
        "checkpoint_path": "",
        "checkpoint_size_bytes": "",
        "latency_status": "missing",
        **{field: "" for field in LATENCY_FIELDS},
    }
    if not run_dir.is_dir():
        return row

    results_path = run_dir / "results.csv"
    checkpoint = run_dir / "weights" / "best.pt"
    required_artifacts = [run_dir / "args.yaml", results_path, checkpoint, run_dir / "weights" / "last.pt"]
    missing = [str(path.relative_to(run_dir)) for path in required_artifacts if not path.is_file()]
    if checkpoint.is_file():
        row["checkpoint_path"] = checkpoint.as_posix()
        row["checkpoint_size_bytes"] = checkpoint.stat().st_size

    best_row, results_error = read_best_validation_row(results_path) if results_path.is_file() else (None, None)
    if best_row is not None:
        epoch = finite_float(best_row.get("epoch"))
        row["best_epoch"] = int(epoch) if epoch is not None and epoch.is_integer() else (epoch if epoch is not None else "")
        for key, column in METRIC_COLUMNS.items():
            value = finite_float(best_row.get(column))
            row[key] = value if value is not None else ""

    run_summary, summary_error = _read_summary(run_dir / "run_summary.json")
    problems = list(missing)
    if results_error:
        problems.append(results_error)
    if summary_error:
        problems.append(summary_error)
    if run_summary is not None:
        recorded_status = run_summary.get("status")
        if recorded_status == "failed":
            problems.append("run_summary records a failed run")
        elif recorded_status != "completed":
            problems.append(f"run status is {run_summary.get('status')!r}, not 'completed'")
        if run_summary.get("model") != model or run_summary.get("seed") != seed:
            problems.append("run_summary model/seed does not match directory name")
        duration = finite_float(run_summary.get("training_duration_seconds"))
        if duration is None or duration < 0:
            problems.append("run_summary has no valid training_duration_seconds")
        else:
            row["training_duration_seconds"] = duration
    if best_row is None:
        problems.append("no usable best validation row")
    absent_metrics = [name for name in METRIC_COLUMNS if row[name] == ""]
    if absent_metrics:
        problems.append("best validation row missing: " + ", ".join(absent_metrics))

    if problems:
        row["run_status"] = "failed" if run_summary is not None and run_summary.get("status") == "failed" else "incomplete"
        row["status_detail"] = "; ".join(problems)
    else:
        row["run_status"] = "completed"
        row["status_detail"] = "completion marker and required artifacts verified"

    latency_path = run_dir / "latency.json"
    if latency_path.is_file() and row["run_status"] == "completed":
        try:
            with latency_path.open("r", encoding="utf-8") as stream:
                latency = json.load(stream)
            latency_matches = (
                latency.get("model") == model
                and latency.get("seed") == seed
                and latency.get("checkpoint_sha256") == sha256_file(checkpoint)
            )
            latency_values = {
                field: finite_float(latency.get("latency_ms", {}).get(source))
                if source != "images_per_second"
                else finite_float(latency.get("effective_images_per_second"))
                for field, source in LATENCY_FIELDS.items()
            }
            if latency_matches and all(value is not None for value in latency_values.values()):
                row["latency_status"] = "available"
                row.update(latency_values)
            else:
                row["latency_status"] = "invalid"
        except (OSError, json.JSONDecodeError, AttributeError):
            row["latency_status"] = "invalid"
    return row


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in ("yolo11s", "yolo26s"):
        model_rows = [row for row in rows if row["model"] == model]
        completed = [row for row in model_rows if row["run_status"] == "completed"]
        summary: dict[str, Any] = {
            "model": model,
            "expected_seeds": "13;21;42",
            "completed_runs": len(completed),
            "summary_status": "complete" if len(completed) == 3 else "incomplete",
            "latency_status": "complete" if sum(row.get("latency_status") == "available" for row in model_rows) == 3 else "incomplete",
        }
        for metric in (*METRIC_COLUMNS.keys(), *LATENCY_FIELDS.keys()):
            eligible = (
                [row for row in model_rows if row["run_status"] == "completed" and row.get("latency_status") == "available"]
                if metric in LATENCY_FIELDS else completed
            )
            values = [finite_float(row[metric]) for row in eligible]
            values = [value for value in values if value is not None]
            if len(eligible) == 3 and len(values) == 3:
                summary[f"{metric}_mean"] = statistics.mean(values)
                summary[f"{metric}_sample_sd"] = statistics.stdev(values)
            else:
                summary[f"{metric}_mean"] = ""
                summary[f"{metric}_sample_sd"] = ""
        output.append(summary)
    return output


def summarize_per_class(
    evaluation_root: Path,
    runs_root: Path,
    completed_pairs: set[tuple[str, int]],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in ("yolo11s", "yolo26s"):
        per_seed: dict[int, dict[int, dict[str, Any]]] = {}
        for seed in (13, 21, 42):
            path = evaluation_root / run_name(model, seed) / "per_class_metrics.csv"
            seed_data: dict[int, dict[str, Any]] = {}
            checkpoint = runs_root / run_name(model, seed) / "weights" / "best.pt"
            validation_summary_path = evaluation_root / run_name(model, seed) / "validation_summary.json"
            valid_evaluation = False
            if (model, seed) in completed_pairs and checkpoint.is_file() and validation_summary_path.is_file():
                try:
                    with validation_summary_path.open("r", encoding="utf-8") as stream:
                        validation_summary = json.load(stream)
                    valid_evaluation = (
                        validation_summary.get("model") == model
                        and validation_summary.get("seed") == seed
                        and validation_summary.get("checkpoint_sha256") == sha256_file(checkpoint)
                    )
                except (OSError, json.JSONDecodeError, AttributeError):
                    valid_evaluation = False
            if valid_evaluation and path.is_file():
                try:
                    with path.open("r", encoding="utf-8-sig", newline="") as stream:
                        for row in csv.DictReader(stream):
                            if row.get("class_status") != "available":
                                continue
                            class_id = int(row["class_id"])
                            if not 0 <= class_id < 32:
                                continue
                            seed_data[class_id] = row
                except (OSError, ValueError, KeyError, csv.Error):
                    seed_data = {}
            per_seed[seed] = seed_data
        for class_id, fdi_code in enumerate(EXPECTED_FDI_CODES):
            record: dict[str, Any] = {
                "model": model,
                "class_id": class_id,
                "class_name": f"FDI_{fdi_code}",
                "observed_seeds": 0,
                "class_summary_status": "incomplete",
            }
            for metric in CLASS_METRICS:
                values = [finite_float(per_seed[seed].get(class_id, {}).get(metric)) for seed in (13, 21, 42)]
                valid = [value for value in values if value is not None]
                record[f"{metric}_mean"] = statistics.mean(valid) if len(valid) == 3 else ""
                record[f"{metric}_sample_sd"] = statistics.stdev(valid) if len(valid) == 3 else ""
                record["observed_seeds"] = min(len(valid), 3) if metric == "ap50_95" else record["observed_seeds"]
            if record["observed_seeds"] == 3:
                record["class_summary_status"] = "complete"
            output.append(record)
    return output


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(
    runs_root: Path,
    output_dir: Path,
    evaluation_root: Path = Path("runs/evaluation"),
) -> tuple[Path, Path, Path]:
    if runs_root.name.lower() != "full":
        raise ValueError("runs root must be a directory named 'full'; pilot results are excluded")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    per_run_rows = [collect_run(runs_root / f"{model}_640_seed{seed}", model, seed) for model, seed in EXPECTED_RUNS]
    summary_rows = summarize_runs(per_run_rows)
    output_dir.mkdir(parents=True, exist_ok=False)
    per_run_path = output_dir / "per_run_results.csv"
    summary_path = output_dir / "model_summary.csv"
    class_summary_path = output_dir / "per_class_summary.csv"
    per_run_fields = [
        "model", "seed", "run_status", "status_detail", "best_epoch", "training_duration_seconds",
        "map50", "map50_95", "precision", "recall", "checkpoint_path", "checkpoint_size_bytes",
        "latency_status", *LATENCY_FIELDS,
    ]
    summary_fields = [
        "model", "expected_seeds", "completed_runs", "summary_status",
        *[f"{metric}_{stat}" for metric in METRIC_COLUMNS for stat in ("mean", "sample_sd")],
        "latency_status",
        *[f"{metric}_{stat}" for metric in LATENCY_FIELDS for stat in ("mean", "sample_sd")],
    ]
    completed_pairs = {
        (row["model"], row["seed"])
        for row in per_run_rows
        if row["run_status"] == "completed"
    }
    class_rows = summarize_per_class(evaluation_root, runs_root, completed_pairs)
    class_fields = [
        "model", "class_id", "class_name", "observed_seeds", "class_summary_status",
        *[f"{metric}_{stat}" for metric in CLASS_METRICS for stat in ("mean", "sample_sd")],
    ]
    write_csv(per_run_path, per_run_rows, per_run_fields)
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(class_summary_path, class_rows, class_fields)
    return per_run_path, summary_path, class_summary_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path("runs/full"), help="Root containing the six full runs (never scans runs/pilot).")
    parser.add_argument("--evaluation-root", type=Path, default=Path("runs/evaluation"), help="Root containing the six per-run validation artifacts.")
    parser.add_argument("--output-dir", type=Path, default=Path("generated/full_experiment_aggregation"), help="New output directory; existing directories are refused.")
    args = parser.parse_args()
    try:
        per_run, summary, class_summary = aggregate(args.runs_root, args.output_dir, args.evaluation_root)
    except (OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(f"Per-run results: {per_run}")
    print(f"Model summary: {summary}")
    print(f"Per-class summary: {class_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
