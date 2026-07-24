"""
Background Engine -- separates the subject from the background.

The final background must be pure black (RGB 0,0,0) with no gray
contamination, so it produces minimal unwanted stippling density.

Not implemented yet -- scaffolding for the next development pass.
"""

from config import settings


class BackgroundEngine:
    """Analyzes and flattens the background region to pure black."""

    def __init__(self, image_path, background_mask):
        self.image_path = image_path
        self.background_mask = background_mask
        self.target_rgb = settings.BACKGROUND_TARGET_RGB

    def analyze(self):
        """Measure background uniformity and separation quality from the subject."""
        raise NotImplementedError("BackgroundEngine.analyze is not implemented yet")

    def process(self):
        """Flatten the background to pure black with a clean subject edge."""
        raise NotImplementedError("BackgroundEngine.process is not implemented yet")

    def export(self, output_dir=None):
        """Write the prepared background layer and analysis data."""
        raise NotImplementedError("BackgroundEngine.export is not implemented yet")
