"""Tests for TonalMapper and DensityEngine."""

import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.density.tonal_mapper import TonalMapper
from modules.density.density_engine import DensityEngine
from config import settings

LENA_PATH = Path(__file__).resolve().parent / "lena.jpg"


class TestTonalMapper(unittest.TestCase):
    def test_produces_exact_level_count_on_gradient(self):
        gray = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        mapper = TonalMapper(levels=32)
        mapper.analyze(gray)
        curve = mapper.process()

        self.assertEqual(len(np.unique(curve)), 32)

    def test_curve_is_monotonic(self):
        gray = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
        mapper = TonalMapper(levels=25)
        mapper.analyze(gray)
        curve = mapper.process()

        self.assertTrue(np.all(np.diff(curve.astype(int)) >= 0))

    def test_handles_flat_image_without_crashing(self):
        gray = np.full((50, 50), 128, dtype=np.uint8)
        mapper = TonalMapper(levels=32)
        mapper.analyze(gray)
        curve = mapper.process()

        self.assertEqual(curve.shape, (256,))

    def test_export_writes_json(self):
        tmp_dir = Path(__file__).resolve().parent / "_tmp_tonal"
        tmp_dir.mkdir(exist_ok=True)
        try:
            gray = np.tile(np.arange(256, dtype=np.uint8), (10, 1))
            mapper = TonalMapper(levels=32)
            mapper.analyze(gray)
            mapper.process()
            path = tmp_dir / "curve.json"
            mapper.export(path)
            self.assertTrue(path.exists())
        finally:
            for p in sorted(tmp_dir.rglob("*"), reverse=True):
                p.unlink()
            tmp_dir.rmdir()


class TestDensityEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_density"
        self.tmp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        self.tmp_dir.rmdir()

    def test_density_levels_within_spec_range(self):
        engine = DensityEngine(LENA_PATH, levels=settings.DENSITY_LEVELS_DEFAULT)
        engine.analyze()
        density, _ = engine.process()

        unique_levels = len(np.unique(density))
        self.assertLessEqual(unique_levels, settings.DENSITY_LEVELS_DEFAULT)
        self.assertGreaterEqual(settings.DENSITY_LEVELS_DEFAULT, settings.DENSITY_LEVELS_MIN)
        self.assertLessEqual(settings.DENSITY_LEVELS_DEFAULT, settings.DENSITY_LEVELS_MAX)

    def test_inverted_map_is_complement(self):
        engine = DensityEngine(LENA_PATH, levels=32)
        density, inverted = engine.process()

        self.assertTrue(np.all(density.astype(int) + inverted.astype(int) == 255))

    def test_does_not_flatten_to_single_value(self):
        engine = DensityEngine(LENA_PATH, levels=32)
        density, _ = engine.process()

        self.assertGreater(len(np.unique(density)), 1, "density map is flat, information destroyed")

    def test_export_writes_maps_and_curve(self):
        engine = DensityEngine(LENA_PATH, levels=32)
        engine.process()
        written = engine.export(self.tmp_dir, person_id="person_01")

        density_dir = self.tmp_dir / "person_01" / "density"
        self.assertTrue((density_dir / settings.FINAL_DENSITY_MAP_NAME).exists())
        self.assertTrue((density_dir / settings.INVERTED_DENSITY_MAP_NAME).exists())
        self.assertTrue((density_dir / "tonal_curve.json").exists())
        self.assertEqual(len(written), 3)

    def test_missing_file_raises(self):
        engine = DensityEngine(self.tmp_dir / "does_not_exist.jpg", levels=32)
        with self.assertRaises(FileNotFoundError):
            engine.analyze()


if __name__ == "__main__":
    unittest.main()
