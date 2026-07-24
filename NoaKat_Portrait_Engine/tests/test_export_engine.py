"""Tests for ExportEngine using synthetic pre-populated masks/ directories."""

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.export.export_engine import ExportEngine
from config import settings


def make_person_density_dir(masks_dir, person_id, levels=32):
    density_dir = masks_dir / person_id / "density"
    density_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(density_dir / settings.FINAL_DENSITY_MAP_NAME), np.full((20, 20), 100, dtype=np.uint8))
    cv2.imwrite(str(density_dir / settings.INVERTED_DENSITY_MAP_NAME), np.full((20, 20), 155, dtype=np.uint8))
    with open(density_dir / "tonal_curve.json", "w") as f:
        json.dump({"levels": levels, "curve": list(range(256))}, f)


class TestExportEngine(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_export"
        self.tmp_dir.mkdir(exist_ok=True)
        self.masks_dir = self.tmp_dir / "masks"

    def tearDown(self):
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        self.tmp_dir.rmdir()

    def test_analyze_raises_without_masks_dir(self):
        engine = ExportEngine(self.tmp_dir)
        with self.assertRaises(FileNotFoundError):
            engine.analyze()

    def test_analyze_finds_single_person(self):
        make_person_density_dir(self.masks_dir, "person_01", levels=32)
        engine = ExportEngine(self.tmp_dir)
        found = engine.analyze()

        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["person_id"], "person_01")
        self.assertEqual(found[0]["levels"], 32)

    def test_analyze_finds_multiple_people_in_order(self):
        make_person_density_dir(self.masks_dir, "person_01")
        make_person_density_dir(self.masks_dir, "person_02")
        engine = ExportEngine(self.tmp_dir)
        found = engine.analyze()

        self.assertEqual([f["person_id"] for f in found], ["person_01", "person_02"])

    def test_process_builds_correct_metadata(self):
        make_person_density_dir(self.masks_dir, "person_01", levels=32)
        make_person_density_dir(self.masks_dir, "person_02", levels=32)
        engine = ExportEngine(self.tmp_dir)
        metadata = engine.process()

        self.assertEqual(metadata, {
            "faces": 2,
            "density_levels": 32,
            "regions": True,
            "stipplism_ready": True,
        })

    def test_export_writes_top_level_files(self):
        make_person_density_dir(self.masks_dir, "person_01", levels=32)
        engine = ExportEngine(self.tmp_dir)
        result = engine.export()

        self.assertTrue(result["final_density_map"].exists())
        self.assertTrue(result["inverted_density_map"].exists())
        self.assertTrue(result["metadata"].exists())

        with open(result["metadata"]) as f:
            data = json.load(f)
        self.assertEqual(data["faces"], 1)
        self.assertTrue(data["stipplism_ready"])

    def test_promoted_density_map_matches_primary_person(self):
        make_person_density_dir(self.masks_dir, "person_01", levels=32)
        engine = ExportEngine(self.tmp_dir)
        engine.export()

        promoted = cv2.imread(str(self.tmp_dir / settings.FINAL_DENSITY_MAP_NAME), cv2.IMREAD_GRAYSCALE)
        original = cv2.imread(
            str(self.masks_dir / "person_01" / "density" / settings.FINAL_DENSITY_MAP_NAME),
            cv2.IMREAD_GRAYSCALE,
        )
        np.testing.assert_array_equal(promoted, original)

    def test_raises_when_no_completed_density_maps(self):
        (self.masks_dir / "person_01").mkdir(parents=True)
        engine = ExportEngine(self.tmp_dir)
        with self.assertRaises(FileNotFoundError):
            engine.analyze()


if __name__ == "__main__":
    unittest.main()
