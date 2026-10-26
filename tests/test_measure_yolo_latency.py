import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from measure_yolo_latency import percentile_nearest_rank


class PercentileTests(unittest.TestCase):
    def test_nearest_rank_percentile(self):
        self.assertEqual(percentile_nearest_rank([5, 1, 4, 2, 3], 95), 5)
        self.assertEqual(percentile_nearest_rank([5, 1, 4, 2, 3], 50), 3)

    def test_rejects_empty_values_and_invalid_percentile(self):
        with self.assertRaises(ValueError):
            percentile_nearest_rank([], 95)
        with self.assertRaises(ValueError):
            percentile_nearest_rank([1], 0)


if __name__ == "__main__":
    unittest.main()
