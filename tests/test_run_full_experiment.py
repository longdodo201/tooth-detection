import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_full_experiment as launcher


class FakeCuda:
    @staticmethod
    def is_available():
        return False


class FakeYOLO:
    fail = False
    received = None

    def __init__(self, checkpoint):
        self.checkpoint = checkpoint

    def train(self, **kwargs):
        type(self).received = kwargs
        if type(self).fail:
            raise RuntimeError("CUDA out of memory")
        run_dir = Path(kwargs["project"]) / kwargs["name"]
        (run_dir / "weights").mkdir()
        (run_dir / "args.yaml").write_text("epochs: 50\n", encoding="utf-8")
        (run_dir / "results.csv").write_text("epoch\n1\n", encoding="utf-8")
        (run_dir / "weights" / "best.pt").write_bytes(b"best")
        (run_dir / "weights" / "last.pt").write_bytes(b"last")


def fake_modules():
    torch = ModuleType("torch")
    torch.__version__ = "test-torch"
    torch.version = SimpleNamespace(cuda="test-cuda")
    torch.cuda = FakeCuda
    torchvision = ModuleType("torchvision")
    torchvision.__version__ = "test-tv"
    ultralytics = ModuleType("ultralytics")
    ultralytics.__version__ = "test-ultralytics"
    ultralytics.YOLO = FakeYOLO
    return {"torch": torch, "torchvision": torchvision, "ultralytics": ultralytics}


class FullExperimentLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / launcher.DATASET
        self.dataset.parent.mkdir(parents=True)
        self.dataset.write_text("path: .\n", encoding="utf-8")
        checkpoint = self.root / launcher.CHECKPOINT_DIR / "yolo11s.pt"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"fake checkpoint")
        self.root_patch = patch.object(launcher, "REPO_ROOT", self.root)
        self.root_patch.start()
        self.cwd_patch = patch("pathlib.Path.cwd", return_value=self.root)
        self.cwd_patch.start()
        FakeYOLO.fail = False
        FakeYOLO.received = None

    def tearDown(self):
        self.cwd_patch.stop()
        self.root_patch.stop()
        self.temp.cleanup()

    def test_success_writes_completion_only_after_required_artifacts(self):
        with patch.dict(sys.modules, fake_modules()):
            run_dir = launcher.run_one("yolo11s", 13)
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["model"], "yolo11s")
        self.assertEqual(summary["seed"], 13)
        self.assertEqual(FakeYOLO.received["batch"], 1)
        self.assertEqual(FakeYOLO.received["fliplr"], 0.0)
        self.assertEqual(FakeYOLO.received["optimizer"], "SGD")
        self.assertTrue((run_dir / "training.log").is_file())

    def test_existing_directory_is_refused_without_overwrite(self):
        run_dir = self.root / "runs" / "full" / "yolo11s_640_seed13"
        run_dir.mkdir(parents=True)
        marker = run_dir / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            launcher.run_one("yolo11s", 13)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_training_failure_records_failed_status_and_oom_without_completion(self):
        FakeYOLO.fail = True
        with patch.dict(sys.modules, fake_modules()):
            with self.assertRaisesRegex(RuntimeError, "out of memory"):
                launcher.run_one("yolo11s", 13)
        run_dir = self.root / "runs" / "full" / "yolo11s_640_seed13"
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["failure_category"], "cuda_oom")
        self.assertFalse((run_dir / "weights" / "best.pt").exists())

    def test_missing_required_artifact_prevents_completed_marker(self):
        class IncompleteYOLO(FakeYOLO):
            def train(self, **kwargs):
                (Path(kwargs["project"]) / kwargs["name"] / "args.yaml").write_text("epochs: 50\n", encoding="utf-8")

        modules = fake_modules()
        modules["ultralytics"].YOLO = IncompleteYOLO
        with patch.dict(sys.modules, modules):
            with self.assertRaisesRegex(RuntimeError, "required artifacts are missing"):
                launcher.run_one("yolo11s", 13)
        run_dir = self.root / "runs" / "full" / "yolo11s_640_seed13"
        summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
        self.assertEqual(summary["status"], "failed")


if __name__ == "__main__":
    unittest.main()
