"""Tests for SkinEngine using a synthetic noisy/textured skin patch."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.skin.skin_engine import SkinEngine


def make_noisy_textured_image():
    """A 200x200 image: skin-toned square (with noise + pore dots) on a flat background."""
    rng = np.random.default_rng(7)
    image = np.full((200, 200, 3), (180, 160, 255), dtype=np.uint8)  # background, BGR
    image[40:160, 40:160] = (140, 170, 220)  # skin-toned square

    skin_mask = np.zeros((200, 200), dtype=np.uint8)
    skin_mask[40:160, 40:160] = 255

    # sensor-noise-like speckle across the whole skin square
    noise = rng.normal(0, 12, size=(120, 120, 3))
    patch = image[40:160, 40:160].astype(np.float64) + noise
    image[40:160, 40:160] = np.clip(patch, 0, 255).astype(np.uint8)

    # regularly spaced "pore" dots -- structured texture that should survive
    for y in range(55, 145, 12):
        for x in range(55, 145, 12):
            cv2.circle(image, (x, y), 2, (100, 130, 180), -1)

    return image, skin_mask


class TestSkinEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_skin"
        self.tmp_dir.mkdir(exist_ok=True)
        self.image, self.skin_mask = make_noisy_textured_image()
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

    def test_reduces_noise_within_skin_region(self):
        engine = SkinEngine(self.image_path, self.skin_mask)
        before = engine.analyze()
        engine.process()

        gray_after = cv2.cvtColor(engine.prepared_image, cv2.COLOR_BGR2GRAY)
        median_after = cv2.medianBlur(gray_after, 3)
        residual_after = cv2.absdiff(gray_after, median_after)[self.skin_mask > 0]
        noise_after = float(np.mean(residual_after))

        self.assertLess(noise_after, before["noise_estimate"])

    def test_preserves_some_texture_variance(self):
        engine = SkinEngine(self.image_path, self.skin_mask)
        engine.process()

        gray_after = cv2.cvtColor(engine.prepared_image, cv2.COLOR_BGR2GRAY)
        std_after = np.std(gray_after[self.skin_mask > 0])
        self.assertGreater(std_after, 3, "skin looks flattened/plastic, not textured")

    def test_leaves_non_skin_pixels_untouched(self):
        engine = SkinEngine(self.image_path, self.skin_mask)
        engine.process()

        outside_original = self.image[self.skin_mask == 0]
        outside_prepared = engine.prepared_image[self.skin_mask == 0]
        np.testing.assert_array_equal(outside_original, outside_prepared)

    def test_export_writes_image_and_analysis(self):
        engine = SkinEngine(self.image_path, self.skin_mask)
        engine.process()
        written = engine.export(self.tmp_dir, person_id="person_01")

        skin_dir = self.tmp_dir / "person_01" / "skin"
        self.assertTrue((skin_dir / "prepared.png").exists())
        self.assertTrue((skin_dir / "analysis.json").exists())
        self.assertEqual(len(written), 2)

    def test_missing_file_raises(self):
        engine = SkinEngine(self.tmp_dir / "does_not_exist.png", self.skin_mask)
        with self.assertRaises(FileNotFoundError):
            engine.analyze()


if __name__ == "__main__":
    unittest.main()
