import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from inspect_dataset import bbox_xywh_to_normalized_xywh, inspect_dataset
from prepare_yolo_labels import build_label_texts, make_fdi_class_mapping, write_label_bundle


def write_minimal_png(path: Path, width: int, height: int) -> None:
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x02\x00\x00\x00"
        + b"\x00\x00\x00\x00"
    )


class BboxConversionTests(unittest.TestCase):
    def test_converts_pixel_xywh_to_normalized_center_xywh(self):
        result = bbox_xywh_to_normalized_xywh([10, 20, 30, 40], 100, 200)
        self.assertEqual(result, (0.25, 0.2, 0.3, 0.2))

    def test_rejects_nonpositive_dimensions(self):
        with self.assertRaises(ValueError):
            bbox_xywh_to_normalized_xywh([0, 0, 1, 1], 0, 20)

    def test_rejects_nonpositive_box_size(self):
        with self.assertRaises(ValueError):
            bbox_xywh_to_normalized_xywh([0, 0, 0, 1], 20, 20)


class DatasetInspectionTests(unittest.TestCase):
    def test_checks_png_dimensions_and_annotation_references(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            write_minimal_png(image_dir / "sample.png", 100, 80)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "sample.png", "width": 100, "height": 80}],
                        "annotations": [
                            {
                                "id": 1,
                                "image_id": 1,
                                "bbox": [10, 20, 30, 40],
                                "category_id_1": 0,
                                "category_id_2": 0,
                            }
                        ],
                        "categories_1": [{"id": 0, "name": 1}],
                        "categories_2": [{"id": 0, "name": "1"}],
                    }
                ),
                encoding="utf-8",
            )
            report, errors = inspect_dataset(annotation_path, image_dir)

        self.assertEqual(errors, [])
        self.assertEqual(report["image_count"], 1)
        self.assertEqual(report["annotation_count"], 1)
        self.assertEqual(report["image_dimensions"], {"100x80": 1})

    def test_reports_bbox_outside_image(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            write_minimal_png(image_dir / "sample.png", 100, 80)
            annotation_path = root / "annotations.json"
            annotation_path.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "sample.png", "width": 100, "height": 80}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "bbox": [90, 20, 20, 10], "category_id_1": 0, "category_id_2": 0}
                        ],
                        "categories_1": [{"id": 0}],
                        "categories_2": [{"id": 0}],
                    }
                ),
                encoding="utf-8",
            )
            _, errors = inspect_dataset(annotation_path, image_dir)

        self.assertTrue(any("outside image bounds" in error for error in errors))


def sample_dataset():
    return {
        "images": [
            {"id": 1, "file_name": "b.png", "width": 100, "height": 80},
            {"id": 2, "file_name": "a.png", "width": 50, "height": 40},
        ],
        "annotations": [
            {"id": 20, "image_id": 1, "bbox": [10, 20, 30, 40], "category_id_1": 41, "category_id_2": 90},
            {"id": 10, "image_id": 1, "bbox": [0, 0, 20, 20], "category_id_1": 40, "category_id_2": 91},
            {"id": 30, "image_id": 2, "bbox": [10, 10, 20, 20], "category_id_1": 40, "category_id_2": 91},
        ],
        "categories_1": [
            {"id": 41, "name": 2}, {"id": 40, "name": 1},
            {"id": 42, "name": 3}, {"id": 43, "name": 4},
        ],
        "categories_2": [
            {"id": 91, "name": "2"}, {"id": 90, "name": "1"},
            {"id": 92, "name": "3"}, {"id": 93, "name": "4"},
            {"id": 94, "name": "5"}, {"id": 95, "name": "6"},
            {"id": 96, "name": "7"}, {"id": 97, "name": "8"},
        ],
    }


class YoloPreparationTests(unittest.TestCase):
    def test_normalizes_using_actual_dimensions_and_rejects_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            write_minimal_png(image_dir / "a.png", 50, 40)
            write_minimal_png(image_dir / "b.png", 100, 80)
            data = sample_dataset()
            class_mapping, _ = make_fdi_class_mapping(data)
            labels = build_label_texts(data, image_dir, class_mapping)

            self.assertEqual(labels["b.txt"].splitlines(), [
                "1 0.10000000 0.12500000 0.20000000 0.25000000",
                "8 0.25000000 0.50000000 0.30000000 0.50000000",
            ])
            data["annotations"][0]["bbox"] = [-1, 0, 20, 20]
            with self.assertRaisesRegex(ValueError, "outside image bounds"):
                build_label_texts(data, image_dir, class_mapping)

    def test_rejects_image_dimension_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            image_dir = Path(directory)
            write_minimal_png(image_dir / "a.png", 51, 40)
            write_minimal_png(image_dir / "b.png", 100, 80)
            data = sample_dataset()
            class_mapping, _ = make_fdi_class_mapping(data)
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                build_label_texts(data, image_dir, class_mapping)

    def test_category_lookup_uses_definition_id_and_name(self):
        data = sample_dataset()
        mapping, metadata = make_fdi_class_mapping(data)
        self.assertEqual(mapping[(40, 91)], 1)  # name 1 x name 2 => FDI 12
        self.assertEqual(mapping[(41, 90)], 8)  # name 2 x name 1 => FDI 21
        entry = metadata["class_to_fdi"][8]
        self.assertEqual(
            {key: entry[key] for key in (
                "fdi_code", "quadrant_number", "tooth_enumeration_number",
                "original_category_id_1", "original_category_id_2",
            )},
            {
                "fdi_code": 21,
                "quadrant_number": 2,
                "tooth_enumeration_number": 1,
                "original_category_id_1": 41,
                "original_category_id_2": 90,
            },
        )

    def test_fdi_class_ids_are_deterministic_and_ordered(self):
        data = sample_dataset()
        mapping, metadata = make_fdi_class_mapping(data)
        self.assertEqual(len(mapping), 32)
        self.assertEqual(metadata["class_to_fdi"][0]["fdi_code"], 11)
        self.assertEqual(metadata["class_to_fdi"][31]["fdi_code"], 48)
        self.assertEqual([entry["class_id"] for entry in metadata["class_to_fdi"]], list(range(32)))
        shuffled = dict(data, annotations=list(reversed(data["annotations"])))
        self.assertEqual(make_fdi_class_mapping(shuffled), (mapping, metadata))

    def test_invalid_category_names_and_id_definitions_are_rejected(self):
        for bad_name in ("one", "5", 1.5, True):
            with self.subTest(name=bad_name):
                data = sample_dataset()
                data["categories_1"][0]["name"] = bad_name
                with self.assertRaises(ValueError):
                    make_fdi_class_mapping(data)

        data = sample_dataset()
        data["categories_2"] = data["categories_2"][:-1]
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            make_fdi_class_mapping(data)

        data = sample_dataset()
        data["categories_1"][0]["category_id"] = 999
        with self.assertRaisesRegex(ValueError, "disagree"):
            make_fdi_class_mapping(data)

    def test_category_id_alias_is_supported(self):
        data = sample_dataset()
        for group in ("categories_1", "categories_2"):
            for category in data[group]:
                category["category_id"] = category.pop("id")
        mapping, _ = make_fdi_class_mapping(data)
        self.assertEqual(mapping[(40, 91)], 1)

    def test_label_text_and_written_bundle_are_deterministic(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "images"
            image_dir.mkdir()
            write_minimal_png(image_dir / "a.png", 50, 40)
            write_minimal_png(image_dir / "b.png", 100, 80)
            data = sample_dataset()
            mapping, metadata = make_fdi_class_mapping(data)
            first = build_label_texts(data, image_dir, mapping)
            shuffled = dict(data, annotations=list(reversed(data["annotations"])))
            second = build_label_texts(shuffled, image_dir, mapping)
            self.assertEqual(first, second)

            output = root / "bundle"
            write_label_bundle(output, first, metadata)
            self.assertEqual((output / "labels" / "b.txt").read_text(encoding="utf-8"), first["b.txt"])
            with self.assertRaises(FileExistsError):
                write_label_bundle(output, second, metadata)


if __name__ == "__main__":
    unittest.main()
