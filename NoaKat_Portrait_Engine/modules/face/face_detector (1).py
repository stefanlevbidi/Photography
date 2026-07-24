"""
Face Detector -- locates every person in a portrait.

Scope: detection only. It finds a bounding box for each face so that
FaceRegions, EyeEngine, SkinEngine, and HairEngine can each work on an
independent person. Supports single portraits today and multi-person group
portraits by design -- detected faces are ordered left-to-right in the
frame and labeled person_01, person_02, ... for stable downstream naming.

Uses OpenCV's bundled Haar cascade classifier (frontal face, vendored under
modules/face/models/) so detection works fully offline with no model
download at run time. This can be swapped for a deep-learning detector
later without changing the analyze/process/export interface.
"""

import json
from pathlib import Path

import cv2

from config import settings

CASCADE_PATH = Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml"


class FaceDetector:
    """Detects faces in an image and assigns each a stable person ID."""

    def __init__(self, image_path, cascade_path=CASCADE_PATH):
        self.image_path = Path(image_path)
        self.cascade_path = Path(cascade_path)
        self.image_size = None
        self.faces = []

    # ------------------------------------------------------------------
    def analyze(self):
        """Run the cascade classifier and return raw face detections."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")

        height, width = image.shape[:2]
        self.image_size = (width, height)

        cascade = cv2.CascadeClassifier(str(self.cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load cascade classifier: {self.cascade_path}")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        boxes, _, weights = cascade.detectMultiScale3(
            gray,
            scaleFactor=1.1,
            minNeighbors=6,
            minSize=(max(30, width // 20), max(30, height // 20)),
            outputRejectLevels=True,
        )

        self.faces = [
            {"x": int(x), "y": int(y), "width": int(w), "height": int(h), "detection_weight": float(weight)}
            for (x, y, w, h), weight in zip(boxes, weights)
        ]
        return self.faces

    # ------------------------------------------------------------------
    def process(self):
        """Order detected faces left-to-right and label them person_01, person_02, ..."""
        if not self.image_size:
            self.analyze()

        ordered = sorted(self.faces, key=lambda face: face["x"])
        width, height = self.image_size

        people = []
        for index, face in enumerate(ordered, start=1):
            box = {"x": face["x"], "y": face["y"], "width": face["width"], "height": face["height"]}
            box_normalized = {
                "x": round(box["x"] / width, 4),
                "y": round(box["y"] / height, 4),
                "width": round(box["width"] / width, 4),
                "height": round(box["height"] / height, 4),
            }
            people.append(
                {
                    "person_id": f"person_{index:02d}",
                    "box": box,
                    "box_normalized": box_normalized,
                    "detection_weight": face["detection_weight"],
                }
            )

        self.faces = people
        return self.faces

    # ------------------------------------------------------------------
    def export(self, output_dir=None):
        """Write each person's face data to output/person_XX/face.json."""
        if not self.faces or "person_id" not in self.faces[0]:
            self.process()

        output_dir = Path(output_dir) if output_dir else settings.OUTPUT_DIR
        written_paths = []
        for person in self.faces:
            person_dir = output_dir / person["person_id"]
            person_dir.mkdir(parents=True, exist_ok=True)
            face_path = person_dir / "face.json"
            with open(face_path, "w") as f:
                json.dump(person, f, indent=2)
            written_paths.append(face_path)

        return written_paths
