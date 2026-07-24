"""Tests for BackgroundEngine using synthetic images and the Lena photo."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.background.background_engine import BackgroundEngine
from modules.face.face_detector import FaceDetector
from modules.face.face_regions import FaceRegions

LENA_PATH = Path(__file__).resolve().parent / "lena.jpg"


class TestBackgroundEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_background"
        self.tmp_dir.mkdir(exist_ok=True)

        # 100x100 image: gray/noisy background, bright subject square in the middle
        rng = np.random.default_rng(3)
        image = np.clip(rng.normal(130, 20, size=(100, 100, 3)), 0, 255).astype(np.uint8)
        image[30:70, 30:70] = (200, 190, 220)
        self.image = image
        self.background_mask = np.full((100, 100), 255, dtype=np.uint8)
        self.background_mask[30:70, 30:70] = 0

        self.image_path = self.tmp_dir / "synthetic.png"
        cv2.imwrite(str(self.image_path), self.image)

    def tearDown(self):
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        self.tmp_dir.rmdir()

    def test_background_becomes_pure_black(self):
        engine = BackgroundEngine(self.image_path, self.background_mask)
        engine.process()

        background_pixels = engine.prepared_image[self.background_mask > 0]
        np.testing.assert_array_equal(background_pixels, 0)

    def test_subject_pixels_untouched(self):
        engine = BackgroundEngine(self.image_path, self.background_mask)
        engine.process()

        subject_original = self.image[self.background_mask == 0]
        subject_prepared = engine.prepared_image[self.background_mask == 0]
        np.testing.assert_array_equal(subject_original, subject_prepared)

    def test_analyze_reports_background_stats(self):
        engine = BackgroundEngine(self.image_path, self.background_mask)
        result = engine.analyze()

        self.assertGreater(result["background_mean"], 0)
        self.assertGreater(result["background_std"], 0)

    def test_export_writes_image_and_analysis(self):
        engine = BackgroundEngine(self.image_path, self.background_mask)
        engine.process()
        written = engine.export(self.tmp_dir, person_id="person_01")

        bg_dir = self.tmp_dir / "person_01" / "background"
        self.assertTrue((bg_dir / "prepared.png").exists())
        self.assertTrue((bg_dir / "analysis.json").exists())
        self.assertEqual(len(written), 2)

        import json
        with open(bg_dir / "analysis.json") as f:
            data = json.load(f)
        self.assertTrue(data["is_pure_black"])
        self.assertEqual(data["background_target_rgb"], [0, 0, 0])

    def test_missing_file_raises(self):
        engine = BackgroundEngine(self.tmp_dir / "does_not_exist.png", self.background_mask)
        with self.assertRaises(FileNotFoundError):
            engine.analyze()

    def test_runs_end_to_end_on_lena(self):
        detector = FaceDetector(LENA_PATH)
        faces = detector.process()
        regions = FaceRegions(LENA_PATH, faces[0]["box"])
        masks = regions.process()

        engine = BackgroundEngine(LENA_PATH, masks["background"])
        engine.process()

        background_pixels = engine.prepared_image[masks["background"] > 0]
        np.testing.assert_array_equal(background_pixels, 0)


if __name__ == "__main__":
    unittest.main()
