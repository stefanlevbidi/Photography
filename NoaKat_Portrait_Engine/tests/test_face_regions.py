"""Tests for FaceRegions using the classic Lena test image."""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.face.face_detector import FaceDetector
from modules.face.face_regions import FaceRegions

LENA_PATH = Path(__file__).resolve().parent / "lena.jpg"


class TestFaceRegions(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp_regions"
        self.tmp_dir.mkdir(exist_ok=True)

        detector = FaceDetector(LENA_PATH)
        faces = detector.process()
        self.face_box = faces[0]["box"]

    def tearDown(self):
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        self.tmp_dir.rmdir()

    def test_detects_two_eyes(self):
        regions = FaceRegions(LENA_PATH, self.face_box)
        regions.analyze()

        self.assertEqual(len(regions.eyes), 2)
        self.assertLess(regions.eyes[0]["x"], regions.eyes[1]["x"])

    def test_process_produces_all_regions_nonempty(self):
        regions = FaceRegions(LENA_PATH, self.face_box)
        masks = regions.process()

        self.assertEqual(set(masks.keys()), set(FaceRegions.REGION_NAMES))
        for name, mask in masks.items():
            self.assertGreater((mask > 0).sum(), 0, f"{name} mask is empty")

    def test_skin_excludes_eyes_and_lips(self):
        regions = FaceRegions(LENA_PATH, self.face_box)
        masks = regions.process()

        overlap_eyes = ((masks["skin"] > 0) & (masks["eyes"] > 0)).sum()
        overlap_lips = ((masks["skin"] > 0) & (masks["lips"] > 0)).sum()
        self.assertEqual(overlap_eyes, 0)
        self.assertEqual(overlap_lips, 0)

    def test_foreground_and_background_do_not_overlap(self):
        regions = FaceRegions(LENA_PATH, self.face_box)
        masks = regions.process()

        overlap = ((masks["background"] > 0) & (regions.foreground_mask > 0)).sum()
        self.assertEqual(overlap, 0)

    def test_export_writes_masks_and_summary(self):
        regions = FaceRegions(LENA_PATH, self.face_box)
        regions.process()
        written_paths = regions.export(self.tmp_dir, person_id="person_01")

        person_dir = self.tmp_dir / "person_01"
        for name in FaceRegions.REGION_NAMES:
            self.assertTrue((person_dir / f"{name}.png").exists())

        summary_path = person_dir / "regions.json"
        self.assertTrue(summary_path.exists())
        with open(summary_path) as f:
            data = json.load(f)
        self.assertEqual(data["person_id"], "person_01")
        self.assertEqual(len(written_paths), len(FaceRegions.REGION_NAMES) + 1)

    def test_missing_file_raises(self):
        regions = FaceRegions(self.tmp_dir / "does_not_exist.jpg", self.face_box)
        with self.assertRaises(FileNotFoundError):
            regions.analyze()


if __name__ == "__main__":
    unittest.main()
