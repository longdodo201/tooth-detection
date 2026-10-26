import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_yolo_checkpoint import per_class_rows
from yolo_eval_common import (
    EXPECTED_FDI_CODES,
    detection_error_counts,
    label_path_for_image,
    load_fdi_names,
    read_yolo_boxes,
    validation_image_refs,
)


class PerClassMetricTests(unittest.TestCase):
    def test_maps_observed_metrics_to_class_ids_and_keeps_unobserved_blank(self):
        metrics = SimpleNamespace(
            ap_class_index=[2, 0],
            p=[0.25, 0.5],
            r=[0.35, 0.6],
            ap50=[0.45, 0.7],
            ap=[0.15, 0.4],
        )
        rows = per_class_rows(metrics, ["FDI_11", "FDI_12", "FDI_13"])
        self.assertEqual(rows[0]["ap50_95"], 0.4)
        self.assertEqual(rows[2]["precision"], 0.25)
        self.assertEqual(rows[1]["class_status"], "not_observed")
        self.assertEqual(rows[1]["ap50"], "")


class DetectionErrorAnalysisTests(unittest.TestCase):
    def test_class_agnostic_greedy_matching_separates_error_types(self):
        true_classes = [0, 1]
        true_boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
        predicted_classes = [1, 1, 2, 3, 4]
        predicted_boxes = [
            (0, 0, 10, 10),     # class confusion against GT 0
            (20, 20, 30, 30),  # correct match against GT 1
            (20, 20, 30, 30),  # duplicate
            (40, 40, 50, 50),  # background FP
            (0, 0, 4, 4),       # overlap below matching IoU
        ]
        counts = detection_error_counts(true_classes, true_boxes, predicted_classes, predicted_boxes, 0.5)
        self.assertEqual(counts[("class_confusion", "0", "1")], 1)
        self.assertEqual(counts[("correct_match", "1", "1")], 1)
        self.assertEqual(counts[("duplicate_prediction", "", "2")], 1)
        self.assertEqual(counts[("background_false_positive", "", "3")], 1)
        self.assertEqual(counts[("localization_error", "", "4")], 1)

    def test_unmatched_ground_truth_is_false_negative(self):
        counts = detection_error_counts([7], [(0, 0, 10, 10)], [], [], 0.5)
        self.assertEqual(counts, {("false_negative", "7", ""): 1})

    def test_matching_rejects_mismatched_class_and_box_lengths(self):
        with self.assertRaises(ValueError):
            detection_error_counts([0], [], [], [])
        with self.assertRaises(ValueError):
            detection_error_counts([], [], [], [], 0)


class LabelReadTests(unittest.TestCase):
    def test_reads_normalized_xywh_as_pixel_xyxy(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            label = Path(temporary) / "image.txt"
            label.write_text("0 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            classes, boxes = read_yolo_boxes(label, 100, 50)
        self.assertEqual(classes, [0])
        self.assertEqual(boxes, [(30.0, 20.0, 70.0, 30.0)])

    def test_rejects_out_of_bounds_and_invalid_labels(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            label = Path(temporary) / "image.txt"
            label.write_text("0 0.1 0.5 0.4 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_yolo_boxes(label, 100, 50)
            label.write_text("32 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                read_yolo_boxes(label, 100, 50)


class ValidationInputTests(unittest.TestCase):
    def test_reads_only_relative_validation_image_manifest_entries(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "images" / "a.png").touch()
            (root / "images" / "b.png").touch()
            (root / "labels" / "a.txt").touch()
            (root / "val.txt").write_text("./images/a.png\nimages/b.png\n", encoding="utf-8")
            refs = validation_image_refs(root, expected_count=2)
            self.assertEqual([ref for ref, _ in refs], ["images/a.png", "images/b.png"])
            self.assertEqual(label_path_for_image(root, refs[0][0]), root / "labels" / "a.txt")

    def test_rejects_manifest_paths_outside_images(self):
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "val.txt").write_text("../test/images/a.png\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                validation_image_refs(root, expected_count=1)

    def test_validates_fdi_class_mapping(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as temporary:
            mapping = Path(temporary) / "class_mapping.json"
            mapping.write_text(json.dumps({
                "class_to_fdi": [
                    {"class_id": index, "fdi_code": code}
                    for index, code in enumerate(EXPECTED_FDI_CODES)
                ]
            }), encoding="utf-8")
            self.assertEqual(load_fdi_names(mapping)[-1], "FDI_48")


if __name__ == "__main__":
    unittest.main()
