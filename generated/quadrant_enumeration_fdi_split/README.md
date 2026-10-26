# Quadrant enumeration FDI train/validation split

This directory contains reproducible split manifests and a YOLO dataset YAML. It does not copy image or label files.

- Strategy: deterministic image-level 80/20 split; seed `42`.
- Train/validation images: 507 / 127.
- Train/validation annotations: 14436 / 3659.
- Group metadata: none was found; patient-level independence cannot be guaranteed.
- Test data was not used.
- Blank label files are accepted and count as zero annotations; this source currently has no blank label files.

The `dataset.yaml` and manifests use the standard YOLO layout. Create local NTFS junctions before using them so `images/` points to the source PNG directory and `labels/` points to the FDI label bundle. The junction paths are ignored by Git and can be recreated from `split_summary.json`:

```powershell
New-Item -ItemType Junction -Path .\images -Target 'D:\long\training_data\quadrant_enumeration\xrays'
New-Item -ItemType Junction -Path .\labels -Target 'D:\long\tooth-detection\generated\quadrant_enumeration_fdi_labels\labels'
```

`train.txt` and `val.txt` list image paths relative to this directory. `dataset.yaml` points to this directory relative to the repository root. Run Ultralytics from the repository root; its `path` is interpreted relative to the process working directory, not the YAML file's location. YOLO resolves labels by replacing the `images` path component with `labels`; no source image or label copies are required.
