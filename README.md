# Tooth Detection

## Current scope

The current experiment direction is bounding-box object detection, with YOLO11s and YOLO26s as candidate models. The label bundle, fixed train/validation split, and a project-local training environment are prepared. Two tiny 640 px pilot runs have completed; full training and model comparison have not started. A safe label-preparation utility is available for validation and optional label-only output after explicit mapping selection.

## Candidate dataset: `quadrant_enumeration`

- Images: 634 PNG files.
- Annotations: 18,095 instances in `train_quadrant_enumeration.json`.
- Each annotation contains `bbox` as `[x, y, width, height]` in image-pixel coordinates, plus a polygon in `segmentation`.
- Image entries include `id`, `file_name`, `width`, and `height`.
- The two category tables contain 4 and 8 numeric labels, respectively; their intended meanings are supplied by the DENTEX description documented below.
- The existing audit is in `reports/annotation_audit/`. Sixteen flagged images remain deferred; the audit flags are not confirmed label errors.

Original images and JSON are external source data and must remain unchanged. Do not copy the full dataset into this repository. Keep test data out of training, validation, tuning, and model selection.

## Confirmed task and FDI mapping

The task is bounding-box tooth detection and enumeration, not disease diagnosis. Per the user-provided DENTEX dataset description, `category_id_1` identifies a quadrant and `category_id_2` identifies tooth enumeration within that quadrant. The FDI code is:

```text
fdi_code = 10 * quadrant_number + tooth_enumeration_number
```

The JSON category definitions use the key `id` (the converter also accepts `category_id`) and a `name`. In the inspected file, `categories_1` has IDs 0, 1, 2, 3 with names 1, 2, 3, 4; `categories_2` has IDs 0–7 with string names `"1"`–`"8"`. The converter looks up each annotation's source category ID in that definition and parses its `name`; it does not treat the raw category ID as a quadrant or tooth number. Expected category names are 1–4 for quadrants and 1–8 for tooth enumeration. Names must parse one-to-one to the complete expected ranges.

The 32 possible FDI codes are ordered numerically: 11–18, 21–28, 31–38, and 41–48. YOLO class IDs are contiguous, zero-based indices in that order. `class_mapping.json` records each YOLO class ID, FDI code, quadrant, tooth enumeration, source category IDs, and source category names.

Dataset limitations: this is an enumerated tooth-label dataset, not a disease-label dataset. The source JSON also contains polygons, but this phase uses bbox labels only. The annotation counts and structural audit do not independently establish clinical correctness. Sixteen previously flagged images remain deferred and are not reviewed in this phase. Do not use test data for training, validation, tuning, or model selection.

## Read-only dataset inspection

The standard-library inspector checks JSON structure, unique image/annotation IDs, image references, PNG dimensions, category-ID references, and bbox values/bounds. It reports the category definitions exactly as stored; it does not infer class semantics or write output files.

```powershell
python scripts/inspect_dataset.py `
  --annotations "D:\path\to\train_quadrant_enumeration.json" `
  --images "D:\path\to\quadrant_enumeration\xrays"
```

Run its unit tests with:

```powershell
python -m unittest discover -s tests -v
```

## YOLO bbox label preparation

`scripts/prepare_yolo_labels.py` validates the source dataset and formats each bbox as a YOLO detection row:

```text
class_id x_center y_center width height
```

Coordinates are normalized using each PNG's actual width and height. Invalid or out-of-bounds boxes are rejected; they are never clipped. The default dry run does not require choosing a class mapping and writes no files. It reports how many label files and rows would be produced while leaving the class mapping unselected:

```powershell
python scripts/prepare_yolo_labels.py `
  --annotations "D:\path\to\train_quadrant_enumeration.json" `
  --images "D:\path\to\quadrant_enumeration\xrays" `
  --dry-run
```

Writing labels requires explicitly selecting `--fdi` and a new output path inside the repository's `generated/` directory. It writes label text and `class_mapping.json`; it does not copy images or create a train/validation split. Existing output directories are never overwritten.

```powershell
python scripts/prepare_yolo_labels.py `
  --annotations "D:\path\to\train_quadrant_enumeration.json" `
  --images "D:\path\to\quadrant_enumeration\xrays" `
  --fdi `
  --dry-run
```

The output is label-only preparation, not a complete training dataset. The converter aborts on invalid category definitions or unexpected category pairs instead of guessing or clipping.

Phase 4 generated the label-only bundle at `generated/quadrant_enumeration_fdi_labels/`. It contains `labels/`, `class_mapping.json`, and a bundle README; source images are not copied, and no split is included.

## Phase 5 train/validation split

`generated/quadrant_enumeration_fdi_split/` contains a reproducible image-level split and `dataset.yaml`. No patient, study, subject, or group identifier was found in the dataset image records or available sidecar metadata. The split therefore uses seed `42` at approximately 80/20 (507 train images, 127 validation images); patient-level independence cannot be guaranteed. All annotations remain with their image. The separate test dataset was not used.

The split root uses local NTFS junctions: `images/` points to the external source PNG directory and `labels/` points to the FDI label bundle. This avoids copying images or labels. The junctions are ignored by Git; recreate them using the paths and commands in the split README if needed. `train.txt` and `val.txt` list paths relative to the split root, and `dataset.yaml` maps class IDs 0–31 to the ordered FDI names.

Blank label files are valid and count as zero objects. None of the 634 current label files is blank. Exact per-class train/validation annotation counts and all source paths are recorded in `split_summary.json`. The split remains vulnerable to patient leakage because patient-level grouping metadata is unavailable.

The dataset YAML `path` is relative to the **repository root working directory**, while `train.txt` and `val.txt` are relative to that dataset root. Run Ultralytics from the repository root (the directory containing this README); running from `generated/quadrant_enumeration_fdi_split/` changes how `path` is interpreted and will fail. The split generator writes this repository-relative path for newly created split outputs.

## Phase 8 640 px pilot

- Environment: Python 3.12.10, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, Ultralytics 8.4.188.
- Both detection checkpoints completed 2 epochs on the same 5% training fraction (25 of 507 train images), at 640 px, batch 1, seed 42, workers 0, AMP enabled, and validation on all 127 validation images.
- Optimizer and augmentation settings were explicitly matched. Horizontal and vertical flips, color changes, mosaic, mixup, copy-paste, geometric transforms, and erasing were disabled for this pilot.
- Logs, arguments, result CSV files, checkpoints, and peak-memory summaries are in the ignored `runs/pilot/` directory. The tiny pilot is only a compatibility and memory check; its metrics are not model-quality evidence.
- The separate 250-image test dataset was not accessed.

## Full experiment launcher and evaluation

The approved full experiment consists of six sequential runs (YOLO11s and YOLO26s, seeds 13/21/42). The launcher uses the existing `.venv`, dataset YAML, and local checkpoints under `.venv/models/`. It writes unique outputs under `runs/full/`; existing paths are refused. A failed run is marked `failed` and blocks automatic retry. After inspecting and manually moving/renaming that failed directory, the same single-run command can be run again. `--all` stops on the first failure.

Run from the repository root:

```powershell
# One run
& .\.venv\Scripts\python.exe scripts\run_full_experiment.py --model yolo11s --seed 13

# All six, sequentially
& .\.venv\Scripts\python.exe scripts\run_full_experiment.py --all

# After all full runs complete, evaluate each best checkpoint on validation
foreach ($model in @('yolo11s', 'yolo26s')) {
    foreach ($seed in @(13, 21, 42)) {
        & .\.venv\Scripts\python.exe scripts\evaluate_yolo_checkpoint.py --model $model --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $model seed $seed" }
    }
}

# After validation evaluation, measure latency per best checkpoint
foreach ($model in @('yolo11s', 'yolo26s')) {
    foreach ($seed in @(13, 21, 42)) {
        & .\.venv\Scripts\python.exe scripts\measure_yolo_latency.py --model $model --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Latency measurement failed: $model seed $seed" }
    }
}

# Aggregate into a new directory (refuses if the destination already exists)
& .\.venv\Scripts\python.exe scripts\aggregate_yolo_results.py `
    --runs-root runs\full `
    --evaluation-root runs\evaluation `
    --output-dir generated\full_experiment_aggregation
```

The default custom error-analysis thresholds are confidence `0.25` and class-agnostic matching IoU `0.50`. These counts are diagnostics, not mAP. Full training, validation evaluation, latency measurement, and aggregation commands are not run as part of repository preparation. See [`docs/full_experiment_protocol.md`](docs/full_experiment_protocol.md) for settings, outputs, and limitations.

## Next phase

The six-run experimental protocol and launcher are ready for user execution. The tiny pilots do not establish full-run convergence or model quality, and patient-level independence remains unverified because group identifiers were not found. Keep the test dataset isolated until its label compatibility is established in a separately approved review.

The approved run protocol and Phase 11 validation, per-class analysis, latency, and aggregation commands are documented in [`docs/full_experiment_protocol.md`](docs/full_experiment_protocol.md). Preparing these tools did not start training.
