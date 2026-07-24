"""
Face Detector -- locates every person in a portrait.

Scope: detection only. It finds bounding boxes/landmarks for each face so
that FaceRegions, EyeEngine, SkinEngine, and HairEngine can each work on an
independent person. Supports single portraits today and multi-person group
portraits by design (person_01, person_02, ...).

Not implemented yet -- scaffolding for the next development pass.
"""


class FaceDetector:
    """Detects faces in an image and assigns each a stable person ID."""

    def __init__(self, image_path):
        self.image_path = image_path
        self.faces = []

    def analyze(self):
        """Locate faces and return landmarks/bounding boxes per person."""
        raise NotImplementedError("FaceDetector.analyze is not implemented yet")

    def process(self):
        """Order and label detected faces as person_01, person_02, ..."""
        raise NotImplementedError("FaceDetector.process is not implemented yet")

    def export(self, output_path=None):
        """Write per-person face data (boxes/landmarks) to JSON."""
        raise NotImplementedError("FaceDetector.export is not implemented yet")
