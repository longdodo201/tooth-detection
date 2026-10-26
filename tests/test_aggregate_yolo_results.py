import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aggregate_yolo_results import aggregate, collect_run, read_best_validation_row, sha256_file, summarize_per_class, summarize_runs


RESULTS_HEADER = [
    "epoch", "time", "metrics/precision(B)", "metrics/recall(B)",
    "metrics/mAP50(B)", "metrics/mAP50-95(B)",
]


def create_run(run_dir: Path, model: str, seed: int, map95: list[float] | None = None) -> None:
    run_dir.mkdir(parents=True)
    (run_dir / "weights").mkdir()
    (run_dir / "args.yaml").write_text("task: detect\n", encoding="utf-8")
    (run_dir / "weights" / "best.pt").write_bytes(b"best")
    (run_dir / "weights" / "last.pt").write_bytes(b"last")
    (run_dir / "run_summary.json").write_text(json.dumps({
        "status": "completed", "model": model, "seed": seed,
        "elapsed_seconds": 13.0, "training_duration_seconds": 12.5,
    }), encoding="utf-8")
    scores = map95 or [0.1, 0.2]
    with (run_dir / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=RESULTS_HEADER)
        writer.writeheader()
        for epoch, map_value in enumerate(scores, start=1):
            writer.writerow({
                "epoch": epoch,
                "time": epoch * 5,
                "metrics/precision(B)": 0.3 + epoch / 100,
                "metrics/recall(B)": 0.4 + epoch / 100,
                "metrics/mAP50(B)": 0.2 + epoch / 100,
                "metrics/mAP50-95(B)": map_value,
            })


class ResultParsingTests(unittest.TestCase):
    def test_selects_highest_map95_and_latest_epoch_on_tie(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=RESULTS_HEADER)
                writer.writeheader()
                for epoch, score in ((1, 0.3), (2, 0.5), (3, 0.5), (4, 0.4)):
                    writer.writerow({key: "0.1" for key in RESULTS_HEADER} | {"epoch": epoch, "metrics/mAP50-95(B)": score})
            row, error = read_best_validation_row(path)
            self.assertIsNone(error)
            self.assertEqual(row["epoch"], "3")

    def test_missing_or_incomplete_run_is_explicit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = collect_run(root / "missing", "yolo11s", 13)
            self.assertEqual(missing["run_status"], "missing_run")
            partial_dir = root / "yolo11s_640_seed13"
            partial_dir.mkdir()
            partial = collect_run(partial_dir, "yolo11s", 13)
            self.assertEqual(partial["run_status"], "incomplete")
            self.assertIn("run_summary.json", partial["status_detail"])

    def test_failed_marker_is_distinguished_from_an_incomplete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "yolo11s_640_seed13"
            run_dir.mkdir()
            (run_dir / "run_summary.json").write_text(json.dumps({
                "status": "failed", "model": "yolo11s", "seed": 13,
                "error_message": "CUDA out of memory",
            }), encoding="utf-8")
            row = collect_run(run_dir, "yolo11s", 13)
            self.assertEqual(row["run_status"], "failed")
            self.assertIn("failed run", row["status_detail"])

    def test_completed_run_records_metrics_epoch_duration_and_checkpoint_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "yolo11s_640_seed13"
            create_run(run_dir, "yolo11s", 13, [0.12, 0.25, 0.21])
            row = collect_run(run_dir, "yolo11s", 13)
            self.assertEqual(row["run_status"], "completed")
            self.assertEqual(row["best_epoch"], 2)
            self.assertEqual(row["training_duration_seconds"], 12.5)
            self.assertEqual(row["map50_95"], 0.25)
            self.assertEqual(row["checkpoint_size_bytes"], 4)

    def test_summary_requires_all_three_completed_seeds_and_uses_sample_sd(self):
        rows = []
        for seed, score in ((13, 0.1), (21, 0.2), (42, 0.3)):
            rows.append({"model": "yolo11s", "seed": seed, "run_status": "completed", "map50": score, "map50_95": score, "precision": score, "recall": score})
        rows.append({"model": "yolo26s", "seed": 13, "run_status": "incomplete", "map50": 0.9, "map50_95": 0.9, "precision": 0.9, "recall": 0.9})
        summary = summarize_runs(rows)
        self.assertEqual(summary[0]["summary_status"], "complete")
        self.assertAlmostEqual(summary[0]["map50_95_mean"], 0.2)
        self.assertAlmostEqual(summary[0]["map50_95_sample_sd"], 0.1)
        self.assertEqual(summary[1]["summary_status"], "incomplete")
        self.assertEqual(summary[1]["map50_95_mean"], "")

    def test_aggregate_refuses_existing_output_and_ignores_other_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs = root / "runs" / "full"
            runs.mkdir(parents=True)
            (root / "runs" / "pilot").mkdir()
            output = root / "generated" / "aggregation"
            per_run, summary, class_summary = aggregate(runs, output)
            with per_run.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 6)
            with summary.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 2)
            with class_summary.open(encoding="utf-8", newline="") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 64)
            with self.assertRaises(FileExistsError):
                aggregate(runs, output)

    def test_per_class_summary_only_computes_statistics_from_three_present_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs_root = root / "runs" / "full"
            evaluation_root = root / "runs" / "evaluation"
            completed_pairs = set()
            for seed, score in ((13, 0.1), (21, 0.2), (42, 0.3)):
                run_dir = runs_root / f"yolo11s_640_seed{seed}"
                create_run(run_dir, "yolo11s", seed)
                checkpoint = run_dir / "weights" / "best.pt"
                eval_dir = evaluation_root / f"yolo11s_640_seed{seed}"
                eval_dir.mkdir(parents=True)
                (eval_dir / "validation_summary.json").write_text(json.dumps({
                    "model": "yolo11s", "seed": seed,
                    "checkpoint_sha256": sha256_file(checkpoint),
                }), encoding="utf-8")
                with (eval_dir / "per_class_metrics.csv").open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["class_id", "class_status", "precision", "recall", "ap50", "ap50_95"])
                    writer.writeheader()
                    writer.writerow({"class_id": 0, "class_status": "available", "precision": score, "recall": score, "ap50": score, "ap50_95": score})
                completed_pairs.add(("yolo11s", seed))
            rows = summarize_per_class(evaluation_root, runs_root, completed_pairs)
            fdi11 = next(row for row in rows if row["model"] == "yolo11s" and row["class_id"] == 0)
            fdi12 = next(row for row in rows if row["model"] == "yolo11s" and row["class_id"] == 1)
            self.assertAlmostEqual(fdi11["ap50_95_mean"], 0.2)
            self.assertAlmostEqual(fdi11["ap50_95_sample_sd"], 0.1)
            self.assertEqual(fdi12["class_summary_status"], "incomplete")
            self.assertEqual(fdi12["ap50_95_mean"], "")


if __name__ == "__main__":
    unittest.main()
