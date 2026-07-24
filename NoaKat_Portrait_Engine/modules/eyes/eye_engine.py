"""
Eye Engine -- professional eye analysis for density preparation.

Locates the left eye and right eye within a face box, then coordinates
IrisEngine and PupilEngine to resolve iris, pupil, and catchlight for
each. Also derives eyelid area (the skin fold directly around the eye
opening) and eye shadow (the socket shading below the eye) from the
detected eye geometry.

Purpose: prepare eye information for the density engine while preserving
expression, depth, and realism. This module never reshapes, enlarges, or
brightens eyes -- it only locates and masks what is already there. It is
not a beauty filter.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from config import settings
from modules.eyes.iris_engine import IrisEngine
from modules.eyes.pupil_engine import PupilEngine

EYE_CASCADE_PATH = Path(__file__).resolve().parent.parent / "face" / "haarcascade_eye.xml"


class EyeEngine:
    """Analyzes and prepares both eyes of a single detected person."""

    def __init__(self, image_path, face_box, eye_cascade_path=EYE_CASCADE_PATH):
        self.image_path = Path(image_path)
        self.face_box = face_box
        self.eye_cascade_path = Path(eye_cascade_path)
        self.image_size = None
        self.eyes = []
        self.masks = {}

    # ------------------------------------------------------------------
    def analyze(self):
        """Detect both eyes and resolve iris/pupil/catchlight for each."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")

        height, width = image.shape[:2]
        self.image_size = (width, height)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        eye_boxes = self._detect_eyes(gray)

        self.eyes = []
        for index, box in enumerate(eye_boxes):
            side = "left" if index == 0 else "right"
            crop = gray[box["y"]:box["y"] + box["height"], box["x"]:box["x"] + box["width"]]

            iris_engine = IrisEngine(crop)
            iris_engine.process()
            pupil_engine = PupilEngine(crop, iris_engine.iris)
            pupil_engine.process()

            self.eyes.append({
                "side": side,
                "box": box,
                "iris": iris_engine.iris,
                "pupil": pupil_engine.pupil,
                "catchlight": pupil_engine.catchlight,
                "_iris_mask": iris_engine.mask,
                "_pupil_mask": pupil_engine.pupil_mask,
                "_catchlight_mask": pupil_engine.catchlight_mask,
            })

        return self.eyes

    def _detect_eyes(self, gray):
        fb = self.face_box
        pad_x, pad_y = int(fb["width"] * 0.1), int(fb["height"] * 0.1)
        rx0 = max(0, fb["x"] - pad_x)
        ry0 = max(0, fb["y"] - pad_y)
        rx1 = min(gray.shape[1], fb["x"] + fb["width"] + pad_x)
        ry1 = min(gray.shape[0], fb["y"] + fb["height"] + pad_y)

        roi_gray = gray[ry0:ry1, rx0:rx1]
        cascade = cv2.CascadeClassifier(str(self.eye_cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load cascade classifier: {self.eye_cascade_path}")

        detections = cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=8, minSize=(20, 20)
        )
        upper_limit = ry0 + int((ry1 - ry0) * 0.66)
        candidates = [
            {"x": int(x + rx0), "y": int(y + ry0), "width": int(w), "height": int(h)}
            for (x, y, w, h) in detections
            if (y + ry0 + h) <= upper_limit
        ]
        candidates.sort(key=lambda e: e["width"] * e["height"], reverse=True)
        top_two = candidates[:2]
        top_two.sort(key=lambda e: e["x"])
        return top_two

    # ------------------------------------------------------------------
    def process(self):
        """Build full-image masks: eyes, iris, pupil, catchlight, eyelid, eye shadow."""
        if not self.eyes:
            self.analyze()

        width, height = self.image_size
        blank = np.zeros((height, width), dtype=np.uint8)
        eyes_mask = blank.copy()
        iris_mask = blank.copy()
        pupil_mask = blank.copy()
        catchlight_mask = blank.copy()
        eyelid_mask = blank.copy()
        eye_shadow_mask = blank.copy()

        for eye in self.eyes:
            box = eye["box"]
            x0, y0 = box["x"], box["y"]

            cv2.rectangle(eyes_mask, (x0, y0), (x0 + box["width"], y0 + box["height"]), 255, -1)
            self._paste_local_mask(iris_mask, eye["_iris_mask"], x0, y0)
            self._paste_local_mask(pupil_mask, eye["_pupil_mask"], x0, y0)
            self._paste_local_mask(catchlight_mask, eye["_catchlight_mask"], x0, y0)

            eyelid_h = int(box["height"] * 0.5)
            eyelid_y0 = max(0, y0 - eyelid_h)
            cv2.rectangle(eyelid_mask, (x0, eyelid_y0), (x0 + box["width"], y0), 255, -1)

            shadow_h = int(box["height"] * 0.4)
            shadow_y1 = min(height, y0 + box["height"] + shadow_h)
            cv2.rectangle(eye_shadow_mask, (x0, y0 + box["height"]), (x0 + box["width"], shadow_y1), 255, -1)

        self.masks = {
            "eyes": eyes_mask,
            "iris": iris_mask,
            "pupil": pupil_mask,
            "catchlight": catchlight_mask,
            "eyelid": eyelid_mask,
            "eye_shadow": eye_shadow_mask,
        }
        return self.masks

    @staticmethod
    def _paste_local_mask(target, local_mask, x0, y0):
        h, w = local_mask.shape[:2]
        region = target[y0:y0 + h, x0:x0 + w]
        target[y0:y0 + h, x0:x0 + w] = cv2.bitwise_or(region, local_mask)

    # ------------------------------------------------------------------
    def export(self, output_dir=None, person_id="person_01"):
        """Write eye masks as PNGs and per-eye analysis as JSON."""
        if not self.masks:
            self.process()

        output_dir = Path(output_dir) if output_dir else settings.MASKS_DIR
        person_dir = output_dir / person_id / "eyes"
        person_dir.mkdir(parents=True, exist_ok=True)

        written_paths = []
        for name, mask in self.masks.items():
            mask_path = person_dir / f"{name}.png"
            cv2.imwrite(str(mask_path), mask)
            written_paths.append(mask_path)

        summary = {
            "person_id": person_id,
            "eyes": [
                {
                    "side": eye["side"],
                    "box": eye["box"],
                    "iris": eye["iris"],
                    "pupil": eye["pupil"],
                    "catchlight": eye["catchlight"],
                }
                for eye in self.eyes
            ],
        }
        summary_path = person_dir / "eyes.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        written_paths.append(summary_path)

        return written_paths
