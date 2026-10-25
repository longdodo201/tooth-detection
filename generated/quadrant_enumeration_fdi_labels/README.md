# FDI YOLO bbox labels

This directory contains label text only, generated from the selected annotation JSON. It does not contain source images, a train/validation split, or model configuration.

Each file in `labels/` uses one YOLO detection row per object: `class_id x_center y_center width height`. Coordinates are normalized to the corresponding source image dimensions. `class_mapping.json` records the FDI code and original category IDs for every contiguous zero-based YOLO class ID. The mapping follows the user-provided DENTEX description and is not a disease classification.
