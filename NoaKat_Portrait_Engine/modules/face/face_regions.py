"""
Face Regions -- builds accurate masks for one detected person.

Produces the masks every other engine consumes: face, skin, eyes,
eyebrows, lips, hair, clothing, background. Masks are per-person so
group portraits keep each subject's regions independent.

Not implemented yet -- scaffolding for the next development pass.
"""


class FaceRegions:
    """Builds segmentation masks for a single detected person."""

    REGION_NAMES = (
        "face", "skin", "eyes", "eyebrows", "lips", "hair", "clothing", "background",
    )

    def __init__(self, image_path, face_landmarks):
        self.image_path = image_path
        self.face_landmarks = face_landmarks
        self.masks = {}

    def analyze(self):
        """Determine region boundaries from face landmarks."""
        raise NotImplementedError("FaceRegions.analyze is not implemented yet")

    def process(self):
        """Rasterize each region into a binary mask."""
        raise NotImplementedError("FaceRegions.process is not implemented yet")

    def export(self, output_dir=None):
        """Write each region mask as a PNG under output/masks/."""
        raise NotImplementedError("FaceRegions.export is not implemented yet")
