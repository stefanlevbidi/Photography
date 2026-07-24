"""Tests for IrisEngine, PupilEngine, and EyeEngine.

Uses a synthetic eye crop (known iris/pupil/catchlight geometry) to verify
exact detection accuracy, and the classic Lena image to verify the full
EyeEngine pipeline runs end-to-end on a real photo.
"""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.eyes.iris_engine import IrisEngine
from modules.eyes.pupil_engine import PupilEngine
from modules.eyes.eye_engine import EyeEngine
from modules.face.face_detector import FaceDetector

LENA_PATH = Path(__file__).resolve().parent / "lena.jpg"


def make_synthetic_eye():
    """A 100x100 crop with a known iris (r=30), pupil (r=12), catchlight (r=4)."""
    crop = np.full((100, 100), 200, dtype=np.uint8)
    cv2.circle(crop, (50, 50), 30, 90, -1)
    cv2.circle(crop, (50, 50), 12, 20, -1)
    cv2.circle(crop, (42, 42), 4, 250, -1)
    return crop


class TestIrisEngine(unittest.TestCase):
    def test_locates_synthetic_iris(self):
        engine = IrisEngine(make_synthetic_eye())
        iris = engine.analyze()

        self.assertAlmostEqual(iris["cx"], 50, delta=3)
        self.assertAlmostEqual(iris["cy"], 50, delta=3)
        self.assertAlmostEqual(iris["r"], 30, delta=4)

    def test_process_returns_mask_matching_crop_size(self):
        crop = make_synthetic_eye()
        engine = IrisEngine(crop)
        mask = engine.process()

        self.assertEqual(mask.shape, crop.shape)
        self.assertGreater((mask > 0).sum(), 0)

    def test_falls_back_on_blank_crop(self):
        blank = np.full((60, 60), 128, dtype=np.uint8)
        engine = IrisEngine(blank)
        iris = engine.analyze()

        self.assertGreater(iris["r"], 0)


class TestPupilEngine(unittest.TestCase):
    def test_locates_synthetic_pupil_and_catchlight(self):
        crop = make_synthetic_eye()
        iris_engine = IrisEngine(crop)
        iris_engine.analyze()

        pupil_engine = PupilEngine(crop, iris_engine.iris)
        result = pupil_engine.analyze()

        self.assertAlmostEqual(result["pupil"]["r"], 12, delta=2)
        self.assertIsNotNone(result["catchlight"])
        self.assertAlmostEqual(result["catchlight"]["cx"], 42, delta=3)
        self.assertAlmostEqual(result["catchlight"]["cy"], 42, delta=3)

    def test_pupil_is_smaller_than_iris(self):
        crop = make_synthetic_eye()
        iris_engine = IrisEngine(crop)
        iris_engine.analyze()

        pupil_engine = PupilEngine(crop, iris_engine.iris)
        pupil_engine.analyze()

        self.assertLess(pupil_engine.pupil["r"], iris_engine.iris["r"])

    def test_no_catchlight_on_flat_iris(self):
        crop = np.full((80, 80), 100, dtype=np.uint8)
        cv2.circle(crop, (40, 40), 25, 60, -1)
        iris_engine = IrisEngine(crop)
        iris_engine.analyze()

        pupil_engine = PupilEngine(crop, iris_engine.iris)
        pupil_engine.analyze()

        self.assertIsNone(pupil_engine.catchlight)


class TestEyeEngine(unittest.TestCase):
    def setUp(self):
        detector = FaceDetector(LENA_PATH)
        faces = detector.process()
        self.face_box = faces[0]["box"]

    def test_detects_both_eyes_with_iris_and_pupil(self):
        engine = EyeEngine(LENA_PATH, self.face_box)
        eyes = engine.analyze()

        self.assertEqual(len(eyes), 2)
        self.assertEqual(eyes[0]["side"], "left")
        self.assertEqual(eyes[1]["side"], "right")
        for eye in eyes:
            self.assertLess(eye["pupil"]["r"], eye["iris"]["r"])

    def test_process_produces_all_masks_nonempty(self):
        engine = EyeEngine(LENA_PATH, self.face_box)
        masks = engine.process()

        expected = {"eyes", "iris", "pupil", "catchlight", "eyelid", "eye_shadow"}
        self.assertEqual(set(masks.keys()), expected)
        for name in ("eyes", "iris", "pupil", "eyelid", "eye_shadow"):
            self.assertGreater((masks[name] > 0).sum(), 0, f"{name} mask is empty")

    def test_export_writes_masks_and_summary(self):
        tmp_dir = Path(__file__).resolve().parent / "_tmp_eyes"
        tmp_dir.mkdir(exist_ok=True)
        try:
            engine = EyeEngine(LENA_PATH, self.face_box)
            engine.process()
            written = engine.export(tmp_dir, person_id="person_01")

            eyes_dir = tmp_dir / "person_01" / "eyes"
            self.assertTrue((eyes_dir / "iris.png").exists())
            self.assertTrue((eyes_dir / "eyes.json").exists())
            self.assertEqual(len(written), 7)
        finally:
            for path in sorted(tmp_dir.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
            for path in sorted(tmp_dir.rglob("*"), reverse=True):
                if path.is_dir():
                    path.rmdir()
            tmp_dir.rmdir()

    def test_missing_file_raises(self):
        engine = EyeEngine(Path("does_not_exist.jpg"), self.face_box)
        with self.assertRaises(FileNotFoundError):
            engine.analyze()


if __name__ == "__main__":
    unittest.main()
