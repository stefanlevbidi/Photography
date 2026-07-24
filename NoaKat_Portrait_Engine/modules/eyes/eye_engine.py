"""
Eye Engine -- professional eye analysis for density preparation.

Scope: locate left eye, right eye, eyelid area, and eye shadow, and
coordinate IrisEngine / PupilEngine for the fine detail inside each eye.
Purpose is to preserve expression, depth, and realism -- this is not a
beauty filter and must never reshape or enhance the eye.

Not implemented yet -- scaffolding for the next development pass.
"""


class EyeEngine:
    """Analyzes and prepares eye regions for the density engine."""

    def __init__(self, image_path, face_regions):
        self.image_path = image_path
        self.face_regions = face_regions
        self.eyes = {}

    def analyze(self):
        """Locate left/right eye, eyelid area, eye shadow, and catchlight."""
        raise NotImplementedError("EyeEngine.analyze is not implemented yet")

    def process(self):
        """Prepare eye tonal detail while preserving expression and depth."""
        raise NotImplementedError("EyeEngine.process is not implemented yet")

    def export(self, output_dir=None):
        """Write eye masks and analysis data."""
        raise NotImplementedError("EyeEngine.export is not implemented yet")
