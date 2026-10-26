import json
import os
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_yolo_split import (
    EXPECTED_FDI_CODES,
    load_fdi_mapping,
    split_image_names,
    validate_images_and_labels,
    write_split_artifacts,
)


def mapping_entries():
    entries = []
    for class_id, code in enumerate(EXPECTED_FDI_CODES):
        entries.append(
            {
                "class_id": class_id,
                "fdi_code": code,
                "quadrant_number": code // 10,
                "tooth_enumeration_number": code % 10,
                "original_category_id_1": code // 10 - 1,
                "original_category_id_2": code % 10 - 1,
            }
        )
    return entries


class DeterministicSplitTests(unittest.TestCase):
    def test_fixed_seed_is_reproducible_and_splits_images_without_overlap(self):
        names = [f"image_{index:03}.png" for index in range(634)]
        train, val = split_image_names(names, seed=42, train_fraction=0.8)
        train_again, val_again = split_image_names(list(reversed(names)), seed=42, train_fraction=0.8)

        self.assertEqual((train, val), (train_again, val_again))
        self.assertEqual((len(train), len(val)), (507, 127))
        self.assertFalse(set(train) & set(val))
        self.assertEqual(set(train) | set(val), set(names))


class SplitInputValidationTests(unittest.TestCase):
    def test_validates_labels_and_accepts_explicit_empty_label_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "positive.png").write_bytes(b"png placeholder")
            (images / "negative.png").write_bytes(b"png placeholder")
            (labels / "positive.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            (labels / "negative.txt").write_text("", encoding="utf-8")

            names, class_counts, rows_by_image = validate_images_and_labels(
                images, labels, mapping_entries()
            )

        self.assertEqual(names, ["negative.png", "positive.png"])
        self.assertEqual(class_counts, Counter({0: 1}))
        self.assertEqual(rows_by_image, {"negative.png": 0, "positive.png": 1})

    def test_rejects_invalid_class_id_and_nonfinite_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "one.png").write_bytes(b"png placeholder")
            label_file = labels / "one.txt"
            label_file.write_text("32 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "class ID"):
                validate_images_and_labels(images, labels, mapping_entries())

            label_file.write_text("0 nan 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_images_and_labels(images, labels, mapping_entries())

            label_file.write_text("0 0.05 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "extends outside image"):
                validate_images_and_labels(images, labels, mapping_entries())

    def test_rejects_missing_label_and_invalid_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            labels = root / "labels"
            images.mkdir()
            labels.mkdir()
            (images / "one.png").write_bytes(b"png placeholder")
            with self.assertRaisesRegex(ValueError, "image/label mismatch"):
                validate_images_and_labels(images, labels, mapping_entries())

            mapping_file = root / "class_mapping.json"
            invalid = mapping_entries()
            invalid[0]["fdi_code"] = 12
            mapping_file.write_text(json.dumps({"class_to_fdi": invalid}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected FDI codes"):
                load_fdi_mapping(mapping_file)


class SplitArtifactTests(unittest.TestCase):
    def test_writes_yaml_and_manifests_and_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "split"
            entries = mapping_entries()
            write_split_artifacts(
                output,
                ["a.png", "b.png"],
                ["c.png"],
                entries,
                2,
                1,
                Counter({0: 2}),
                Counter({1: 1}),
                42,
                0.8,
                root / "source_images",
                root / "source_labels",
                root / "class_mapping.json",
            )
            self.assertEqual((output / "train.txt").read_text(encoding="utf-8"), "./images/a.png\n./images/b.png\n")
            self.assertEqual((output / "val.txt").read_text(encoding="utf-8"), "./images/c.png\n")
            yaml_text = (output / "dataset.yaml").read_text(encoding="utf-8")
            try:
                expected_path = os.path.relpath(
                    output.resolve(), start=Path(__file__).resolve().parents[1]
                ).replace("\\", "/")
            except ValueError:
                expected_path = output.resolve().as_posix()
            self.assertIn(f"path: {expected_path}\ntrain: train.txt\nval: val.txt", yaml_text)
            self.assertIn("31: FDI_48", yaml_text)
            self.assertIn("group_independence_guaranteed\": false", (output / "split_summary.json").read_text(encoding="utf-8"))
            with self.assertRaises(FileExistsError):
                write_split_artifacts(
                    output,
                    [], [], entries, 0, 0, Counter(), Counter(), 42, 0.8,
                    root, root, root,
                )


if __name__ == "__main__":
    unittest.main()
