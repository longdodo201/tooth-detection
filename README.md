# Tooth Detection

## Current scope

The current experiment direction is bounding-box object detection, with YOLO11s and YOLO26s as candidate models. Training and training-ready dataset creation have not started. A safe label-preparation utility is available for validation and optional label-only output after explicit mapping selection.

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

## Next steps after semantics are confirmed

1. Record the target definition and class mapping with its source.
2. Check for patient/group identifiers; if available, keep groups together in a deterministic train/validation split.
3. Convert a copy of the labels to the chosen YOLO detection format and validate the conversion.
4. Keep test data isolated for final evaluation only.
5. Verify both model variants, then compare them with the same split, image size, protocol, and comparable training budget.

Do not install packages, download model weights, train, or make Git commits/pushes without the user's direction.
