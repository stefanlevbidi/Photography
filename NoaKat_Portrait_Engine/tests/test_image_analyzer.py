"""Tests for ImageAnalyzer using synthetic generated images (no fixtures needed)."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.analyzer.image_analyzer import ImageAnalyzer


class TestImageAnalyzer(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp"
        self.tmp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for path in self.tmp_dir.glob("*"):
            path.unlink()
        self.tmp_dir.rmdir()

    def _make_image(self, name, array, mode="RGB"):
        path = self.tmp_dir / name
        Image.fromarray(array, mode=mode).save(path)
        return path

    def test_normal_midtone_image_has_no_exposure_problems(self):
        rng = np.random.default_rng(42)
        gray = np.clip(rng.normal(loc=128, scale=50, size=(1600, 1600)), 0, 255)
        array = np.repeat(gray[:, :, np.newaxis], 3, axis=2).astype(np.uint8)
        path = self._make_image("normal.png", array)

        analyzer = ImageAnalyzer(path)
        result = analyzer.process()

        self.assertEqual(result["resolution"]["width"], 1600)
        self.assertEqual(result["resolution"]["height"], 1600)
        self.assertEqual(result["color_mode"]["mode"], "RGB")
        self.assertFalse(result["color_mode"]["is_grayscale"])
        self.assertEqual(result["exposure_problems"]["problems"], [])
        self.assertTrue(result["is_ready_for_pipeline"])

    def test_underexposed_flat_image_is_flagged(self):
        array = np.full((100, 100, 3), 20, dtype=np.uint8)
        path = self._make_image("dark.png", array)

        analyzer = ImageAnalyzer(path)
        result = analyzer.process()

        self.assertIn("underexposed", result["exposure_problems"]["problems"])
        self.assertIn("flat_low_contrast", result["exposure_problems"]["problems"])
        self.assertIn("resolution_below_recommended_minimum", result["exposure_problems"]["problems"])
        self.assertFalse(result["is_ready_for_pipeline"])

    def test_overexposed_image_is_flagged(self):
        array = np.full((1600, 1600, 3), 254, dtype=np.uint8)
        path = self._make_image("bright.png", array)

        analyzer = ImageAnalyzer(path)
        result = analyzer.process()

        self.assertIn("overexposed", result["exposure_problems"]["problems"])
        self.assertIn("clipped_highlights", result["exposure_problems"]["problems"])

    def test_histogram_sums_to_total_pixels(self):
        array = np.full((50, 60, 3), 100, dtype=np.uint8)
        path = self._make_image("uniform.png", array)

        analyzer = ImageAnalyzer(path)
        result = analyzer.analyze()

        self.assertEqual(sum(result["histogram"]["counts"]), 50 * 60)

    def test_export_writes_valid_json(self):
        array = np.full((1600, 1600, 3), 128, dtype=np.uint8)
        path = self._make_image("export_me.png", array)

        analyzer = ImageAnalyzer(path)
        output_path = self.tmp_dir / "analysis.json"
        analyzer.export(output_path)

        self.assertTrue(output_path.exists())
        with open(output_path) as f:
            data = json.load(f)
        self.assertIn("brightness", data)
        self.assertIn("is_ready_for_pipeline", data)

    def test_missing_file_raises(self):
        analyzer = ImageAnalyzer(self.tmp_dir / "does_not_exist.png")
        with self.assertRaises(FileNotFoundError):
            analyzer.analyze()


if __name__ == "__main__":
    unittest.main()
