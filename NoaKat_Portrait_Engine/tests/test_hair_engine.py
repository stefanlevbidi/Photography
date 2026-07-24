"""Tests for HairEngine using synthetic directional/textured patches and Lena."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.hair.hair_engine import HairEngine
from modules.face.face_detector import FaceDetector
from modules.face.face_regions import FaceRegions

LENA_PATH = Path(__file__).resolve().parent / "lena.jpg"


def make_striped_image(angle_lines):
    """A 200x200 image with parallel stripes, plus a dark half and bright half."""
    image = np.full((200, 200, 3), 120, dtype=np.uint8)
    for i in range(-200, 200, 10):
        cv2.line(image, (i, 0), (i + angle_lines, 200), (200, 200, 200), 3)
    image[:, :100] = np.clip(image[:, :100].astype(int) - 80, 0, 255).astype(np.uint8)
    image[:, 100:] = np.clip(image[:, 100:].astype(int) + 60, 0, 255).astype(np.uint8)
    mask = np.full((200, 200), 255, dtype=np.uint8)
    return image, mask


class TestHairEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_hair"
        self.tmp_dir.mkdir(exist_ok=True)
        self.image, self.mask = make_striped_image(angle_lines=200)  # ~45 degrees
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

    def test_estimates_dominant_direction(self):
        engine = HairEngine(self.image_path, self.mask)
        result = engine.analyze()

        self.assertIsNotNone(result["dominant_angle_degrees"])
        self.assertAlmostEqual(result["dominant_angle_degrees"], 45, delta=5)

    def test_measures_dark_and_highlight_ratios(self):
        engine = HairEngine(self.image_path, self.mask)
        result = engine.analyze()

        self.assertGreater(result["dark_mass_ratio"], 0)
        self.assertGreater(result["highlight_ratio"], 0)

    def test_no_direction_on_empty_mask(self):
        empty_mask = np.zeros((200, 200), dtype=np.uint8)
        engine = HairEngine(self.image_path, empty_mask)
        result = engine.analyze()

        self.assertIsNone(result["dominant_angle_degrees"])

    def test_leaves_non_hair_pixels_untouched(self):
        half_mask = np.zeros((200, 200), dtype=np.uint8)
        half_mask[:, 100:] = 255
        engine = HairEngine(self.image_path, half_mask)
        engine.process()

        outside_original = self.image[half_mask == 0]
        outside_prepared = engine.prepared_image[half_mask == 0]
        np.testing.assert_array_equal(outside_original, outside_prepared)

    def test_export_writes_image_and_analysis(self):
        engine = HairEngine(self.image_path, self.mask)
        engine.process()
        written = engine.export(self.tmp_dir, person_id="person_01")

        hair_dir = self.tmp_dir / "person_01" / "hair"
        self.assertTrue((hair_dir / "prepared.png").exists())
        self.assertTrue((hair_dir / "analysis.json").exists())
        self.assertEqual(len(written), 2)

    def test_missing_file_raises(self):
        engine = HairEngine(self.tmp_dir / "does_not_exist.png", self.mask)
        with self.assertRaises(FileNotFoundError):
            engine.analyze()

    def test_runs_end_to_end_on_lena_hair_region(self):
        detector = FaceDetector(LENA_PATH)
        faces = detector.process()
        regions = FaceRegions(LENA_PATH, faces[0]["box"])
        masks = regions.process()

        engine = HairEngine(LENA_PATH, masks["hair"])
        result = engine.analyze()
        engine.process()

        self.assertGreaterEqual(result["dark_mass_ratio"], 0)
        self.assertEqual(engine.prepared_image.shape, engine.image.shape)


if __name__ == "__main__":
    unittest.main()
