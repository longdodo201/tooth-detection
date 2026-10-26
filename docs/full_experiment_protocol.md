# Full Experiment Protocol

Status: **finalized protocol for review; no training has been run under this protocol**.

## Objective and fixed data

Compare YOLO11s and YOLO26s for bounding-box detection of 32 FDI tooth positions in `quadrant_enumeration`. The task is tooth position/enumeration detection, not disease diagnosis.

Use the existing image-level split without changes: 507 train images (14,436 objects), 127 validation images (3,659 objects), split seed `42`. Use the existing class mapping, labels, manifests, and dataset YAML at `generated/quadrant_enumeration_fdi_split/dataset.yaml`. Run commands from the repository root because the YAML `path` is repository-relative. The layout requires the local `images/` and `labels/` junctions documented in the split README. Blank label files mean zero objects and are supported; none are present in this split.

The separate test set is described in the project context as disease-labelled. Its compatibility with 32-class FDI detection has not been established. Do not access, load, copy, inspect, or evaluate it; this protocol does not claim test evaluation is ready. The split is image-level because no reliable patient, study, or group identifier was found in available repository dataset records/sidecar metadata. Patient-level leakage therefore remains possible.

## Approved experiment design

Run six independent fine-tuning jobs: two models × three training seeds. Pair each model on each same seed so stochastic initializations are comparable. The split seed stays `42`; the listed training seeds do not regenerate the split.

| Setting | Value | Evidence / note |
|---|---:|---|
| Models | YOLO11s, YOLO26s | Candidate models; both completed the Phase 8 pilot |
| Training seeds | `13`, `21`, `42` for each model | Approved; six runs total |
| Split | Existing 507/127 image-level split, split seed `42` | Fixed; do not regenerate |
| Image size / batch | `640` / `1` | Approved; batch 1 is conservative for RTX 2050 4 GB, but full-run compatibility has not been demonstrated |
| Epochs / early stopping | Maximum `50`, patience `20` | Approved; patience monitors validation fitness (see checkpoint section) |
| Optimizer | Explicit `SGD` for both | Approved; same settings where supported |
| Validation | Run each epoch on the existing validation split | Validation only for checkpoint selection and reporting |
| Outputs | One directory per model and seed under `runs/full/` | Unique names; never use `runs/pilot/` or overwrite an existing run |

The Phase 8 pilot used Python 3.12.10, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, Ultralytics 8.4.188, and an NVIDIA GeForce RTX 2050. It completed both models at 640 px and batch 1, but only for two epochs on 5% of the training images. Pilot metrics are not model-comparison results and must not be reported as final evidence.

### Matched training arguments

Use the same explicit arguments in all six runs where the installed version supports them:

- Pretrained local checkpoint, `pretrained=True`, `task='detect'`, `fraction=1.0`.
- `seed` set to the current run's approved seed; `deterministic=True`.
- `optimizer='SGD'`, `lr0=0.001`, `lrf=0.01`, `momentum=0.937`, `weight_decay=0.0005`, `nbs=1`.
- `cos_lr=False`: the installed configuration uses the linear learning-rate schedule from `lr0` toward `lr0 * lrf` over the epoch budget.
- `warmup_epochs=0.0`; warm-up is disabled. The pilot's `warmup_momentum=0.8` and `warmup_bias_lr=0.1` are inactive when warm-up epochs are zero; keep them recorded as resolved defaults.
- `workers=0`, `cache=False`, `amp=True`, `device=0`, `val=True`, `plots=True`, `save=True`, `exist_ok=False`.
- Disable the pilot's augmentations explicitly: `hsv_h=0`, `hsv_s=0`, `hsv_v=0`, `degrees=0`, `translate=0`, `scale=0`, `shear=0`, `perspective=0`, `flipud=0`, `fliplr=0`, `bgr=0`, `mosaic=0`, `mixup=0`, `copy_paste=0`, `cutmix=0`, `auto_augment=None`, `erasing=0`, and `close_mosaic=0`. Horizontal flipping stays disabled to avoid spatially reversing left/right FDI labels.

The SGD, learning-rate, warm-up, and disabled-augmentation values are fixed for this experiment as specified by the user; the pilot verifies that these arguments ran, not that they are optimal. The installed version's defaults also include model-specific loss/head behavior; identical public arguments do not make the two architectures' optimization or compute identical. Preserve the resolved `args.yaml` for every run and report this limitation.

### Checkpoint selection and installed-version verification

The active `.venv` contains Python 3.12.10, PyTorch 2.7.0+cu128, torchvision 0.22.0+cu128, and Ultralytics 8.4.188. The installed Ultralytics configuration supports `epochs`, `patience`, `seed`, `deterministic`, and the explicit SGD/LR arguments. Source inspection of this installed version shows that detection metrics' fitness is exactly box mAP@0.5:0.95; the trainer's early stopper and `best.pt` selection use that fitness. Thus `best.pt` is the approved highest-validation-mAP@0.5:0.95 checkpoint for this pinned version. `last.pt` is also retained. If Ultralytics is upgraded, recheck this behavior before relying on `best.pt`.

Record the selected epoch and validation mAP@0.5:0.95. If multiple epochs tie for the maximum, Ultralytics 8.4.188 writes the latest tied epoch to `best.pt` (its save logic writes when the current fitness equals the best fitness). Record exact package versions and the complete `args.yaml` with each run.

### Training launcher

`scripts/run_full_experiment.py` is the supported launcher. Run it from the repository root with `.venv`; it checks the fixed dataset YAML and local checkpoints in `.venv/models/`, creates only the six exact run names under `runs/full/`, runs sequentially, and refuses any existing run directory (including a prior failure). It does not download missing checkpoints or alter the batch/settings after an OOM. `training.log`, Ultralytics' `args.yaml`, `results.csv`, and `weights/best.pt`/`last.pt` are retained. `run_summary.json` is marked `completed` only after all required outputs exist; failures get a `failed` marker and error details.

Run one model/seed or all six:

```powershell
& .\.venv\Scripts\python.exe scripts\run_full_experiment.py --model yolo11s --seed 13
& .\.venv\Scripts\python.exe scripts\run_full_experiment.py --all
```

To continue after a failed run, inspect its log and `run_summary.json`, then move or rename that run directory manually after deciding how to preserve it. Rerun the same one model/seed command; the launcher never retries automatically or overwrites it. For sequential `--all`, the first failure stops later runs; fix or archive the failed output before starting `--all` again.

## Phase 11 post-training tools

The following tools have been added and are not run as part of this phase. Run commands from the repository root using `.venv`. They address only the six exact names under `runs/full/`, and validation tools read `generated/quadrant_enumeration_fdi_split/val.txt` (127 images). They do not use the held-out test data.

### Validation metrics and custom error counts

Run once for each completed run. Each command requires a completed `run_summary.json`, `weights/best.pt`, and the existing validation junctions. The CLI defaults to confidence `0.25` and match IoU `0.50`; both may be supplied explicitly as shown. It refuses an existing evaluation directory.

```powershell
$python = '.\.venv\Scripts\python.exe'
foreach ($model in @('yolo11s', 'yolo26s')) {
    foreach ($seed in @(13, 21, 42)) {
        & $python scripts\evaluate_yolo_checkpoint.py `
            --model $model --seed $seed `
            --error-confidence 0.25 --error-match-iou 0.50
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: $model seed $seed" }
    }
}
```

The command creates `runs/evaluation/<model>_640_seed<seed>/` containing Ultralytics validation results/plots, `validation_summary.json`, `per_class_metrics.csv`, `detection_error_analysis.csv`, and `detection_error_analysis.json`. Ultralytics standard detection metrics are mAP@0.5, mAP@0.5:0.95, precision, recall, and its built-in confusion-matrix plots. `DetMetrics.box.ap_class_index`, `.ap50`, `.ap`, `.p`, and `.r` supply the per-class values. Precision and recall are taken from the installed Ultralytics operating point selected for maximum mean F1, not from the custom error-analysis confidence threshold. Unobserved classes remain blank and marked `not_observed`; they are not filled with an aggregate AP.

The custom error CSV is diagnostic and is **not mAP**. It uses confidence `0.25` and class-agnostic greedy one-to-one matching, considering all ground-truth/prediction pairs at or above IoU `0.50`, sorted by descending IoU (ties use ground-truth then prediction order). It counts correct matches, class confusions, false negatives, duplicate predictions, localization errors (unmatched prediction with positive but sub-threshold overlap), and background false positives. Prediction NMS IoU is `0.70`, separate from the matching IoU. These fixed values are recorded in each error-analysis artifact and must not be changed after inspecting final results.

### Inference latency

Run for every completed best checkpoint. The utility first decodes all 127 validation images into memory; timing starts after loading the model and excludes image disk decoding/loading. It warms up on the first 20 images, then measures three sequential passes through all 127 images. Each timed call uses batch 1, 640 px, CUDA device 0, FP32 (`quantize=32` in the installed Ultralytics predict API) and CUDA synchronization immediately before and after prediction. Reported total latency includes in-memory Ultralytics preprocessing, model execution, postprocessing, and Python call overhead. It writes mean, median, nearest-rank p95, effective images/second, Ultralytics timing components when available, checkpoint hash/size, GPU name/memory/driver, and software versions to a new `latency.json`; an existing file is refused.

```powershell
$python = '.\.venv\Scripts\python.exe'
foreach ($model in @('yolo11s', 'yolo26s')) {
    foreach ($seed in @(13, 21, 42)) {
        & $python scripts\measure_yolo_latency.py --model $model --seed $seed
        if ($LASTEXITCODE -ne 0) { throw "Latency measurement failed: $model seed $seed" }
    }
}
```

Latency is measured per checkpoint; summarize its three run reports with mean and sample standard deviation. Use the same RTX 2050 and record driver/software details for all six measurements. Do not use the pilot checkpoints.

### CSV aggregation

After training and any per-run evaluation/latency measurements, create a new aggregation directory. The default command refuses to overwrite the destination:

```powershell
& .\.venv\Scripts\python.exe scripts\aggregate_yolo_results.py `
    --runs-root runs\full `
    --evaluation-root runs\evaluation `
    --output-dir generated\full_experiment_aggregation
```

Outputs:

- `per_run_results.csv`: exactly one row for each of the six expected model/seed pairs; missing runs, incomplete artifacts, best epoch, duration marker, validation metrics, checkpoint path/size, and latency status are explicit.
- `model_summary.csv`: per-model mean and sample SD for validation metrics only when all three seeds have completed runs; latency mean/SD only when all three matching latency reports are available. Missing values are blank, and the status is `incomplete` rather than an estimate from fewer seeds.
- `per_class_summary.csv`: per-model/per-class mean and sample SD for precision, recall, AP50, and AP50-95 only when all three per-run validation CSVs contain that class/metric; otherwise values stay blank with the observed-seed count.

The aggregator reads only the six expected directories under `runs/full/`, never `runs/pilot/`. It selects the highest mAP@0.5:0.95 row from each `results.csv` (latest epoch on ties), reports that row's corresponding mAP50/precision/recall, and obtains duration/completion from `run_summary.json`. A completed run requires `args.yaml`, `results.csv`, `weights/best.pt`, `weights/last.pt`, and a matching successful completion marker. It distinguishes failed from incomplete runs, excludes both from summaries, and accepts per-class/latency evaluation artifacts only when their model, seed, and checkpoint hash match a completed official run. It never synthesizes missing metrics or pools predictions.

## Evaluation and results

Report each run separately, then summarize each model across its three seeds using the mean and sample standard deviation (`n=3`). For primary metrics, report per-run values and model-level mean ± sample SD. Report per-class AP for all 32 FDI classes per run and summarize each class across seeds. Do not merge or pool predictions from different checkpoints as if they came from one model.

Required results for each selected `best.pt`:

- Validation mAP@0.5, mAP@0.5:0.95, precision, and recall.
- Per-class AP/results for FDI 11–18, 21–28, 31–38, and 41–48.
- Detection error analysis: class confusions among matched boxes, false positives, false negatives, and localization errors. A conventional classification confusion matrix alone is not sufficient for detection.
- Epoch at stopping, epoch selected, training duration, checkpoint size, resolved arguments, log/results paths, and best/last checkpoints.
- Python/package versions, OS, GPU and driver, device, AMP/precision, checkpoint source/path and SHA-256, split seed, training seed, and dataset YAML identity.

Use validation only for training-time checkpoint selection and final validation reporting. The installed `model.val` command can report the selected checkpoint's aggregate and per-class results; a future reproducible evaluation invocation should use `split='val'`, `imgsz=640`, `batch=1`, `device=0`, and a unique output directory per run.

### Latency and error-analysis caveats

The in-memory latency utility uses only validation images and separates disk decode/load from timed prediction calls; timing still includes preprocessing, postprocessing, and Python call overhead. It specifies FP32 with the installed predict API's `quantize=32` option. The custom error counts are threshold-dependent diagnostic summaries, not standard object-detection mAP. The agreed thresholds are confidence `0.25` and matching IoU `0.50`.

## Data isolation and limitations

Do not access, load, copy, inspect, or evaluate test files. Test-label compatibility with the 32-class FDI task is unestablished; no test evaluation is ready. The image-level split cannot rule out patient-level leakage. Keep the split fixed as approved.

Do not interpret pilot metrics as final results. Do not pool predictions across seeds. Matching the explicit settings controls the experiment inputs but cannot erase differences in model architecture, loss functions, parameter counts, or optimization behavior.

## Artifacts and remaining implementation work

Each `runs/full/<model>_640_seed<seed>/` directory should retain `args.yaml`, logs, `results.csv`, plots, validation artifacts, `weights/best.pt`, `weights/last.pt`, and timing/metrics summaries. Record checkpoint sizes and provenance alongside the run. Never write into `runs/pilot/` or reuse a run directory.

The repository launcher and Phase 11 validation/per-class/error analysis, latency measurement, and aggregation utilities are prepared for future use. No full-run training, checkpoint validation, latency measurement, or result aggregation has been executed by preparing these tools.

No training, tuning, test evaluation, split generation, label generation, package installation, checkpoint download, or Git history operation is part of this protocol-finalization phase.
