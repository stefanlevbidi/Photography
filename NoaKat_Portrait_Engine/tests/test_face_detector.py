"""Tests for FaceDetector using the classic Lena test image and synthetic composites."""

import json
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from modules.face.face_detector import FaceDetector

LENA_PATH = Path(__file__).resolve().parent / "lena.jpg"


class TestFaceDetector(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(__file__).resolve().parent / "_tmp"
        self.tmp_dir.mkdir(exist_ok=True)

    def tearDown(self):
        for path in self.tmp_dir.rglob("*"):
            if path.is_file():
                path.unlink()
        for path in sorted(self.tmp_dir.rglob("*"), reverse=True):
            if path.is_dir():
                path.rmdir()
        self.tmp_dir.rmdir()

    def test_detects_single_face(self):
        detector = FaceDetector(LENA_PATH)
        faces = detector.process()

        self.assertEqual(len(faces), 1)
        self.assertEqual(faces[0]["person_id"], "person_01")
        self.assertIn("box", faces[0])
        self.assertIn("box_normalized", faces[0])

    def test_no_faces_in_blank_image(self):
        blank_path = self.tmp_dir / "blank.png"
        array = np.full((400, 400, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(blank_path), array)

        detector = FaceDetector(blank_path)
        faces = detector.process()

        self.assertEqual(faces, [])

    def test_multiple_faces_ordered_left_to_right(self):
        lena = cv2.imread(str(LENA_PATH))
        canvas = np.full((lena.shape[0], lena.shape[1] * 2 + 100, 3), 128, dtype=np.uint8)
        canvas[:, : lena.shape[1]] = lena
        canvas[:, lena.shape[1] + 100 :] = lena
        composite_path = self.tmp_dir / "two_faces.png"
        cv2.imwrite(str(composite_path), canvas)

        detector = FaceDetector(composite_path)
        faces = detector.process()

        self.assertEqual(len(faces), 2)
        self.assertEqual([f["person_id"] for f in faces], ["person_01", "person_02"])
        self.assertLess(faces[0]["box"]["x"], faces[1]["box"]["x"])

    def test_export_writes_per_person_json(self):
        detector = FaceDetector(LENA_PATH)
        detector.process()
        written_paths = detector.export(self.tmp_dir)

        self.assertEqual(len(written_paths), 1)
        person_01_path = self.tmp_dir / "person_01" / "face.json"
        self.assertTrue(person_01_path.exists())
        with open(person_01_path) as f:
            data = json.load(f)
        self.assertEqual(data["person_id"], "person_01")

    def test_missing_file_raises(self):
        detector = FaceDetector(self.tmp_dir / "does_not_exist.jpg")
        with self.assertRaises(FileNotFoundError):
            detector.analyze()


if __name__ == "__main__":
    unittest.main()
