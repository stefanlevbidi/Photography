"""
Skin Engine -- prepares skin for stippling without faking it.

Keeps natural texture, pores, and shadow modeling. Removes sensor noise
and compression artifacts only. Must never produce plastic, blurred, or
artificially smoothed skin -- that destroys the density information
stippling depends on.

Not implemented yet -- scaffolding for the next development pass.
"""


class SkinEngine:
    """Analyzes and prepares the skin region for density mapping."""

    def __init__(self, image_path, skin_mask):
        self.image_path = image_path
        self.skin_mask = skin_mask

    def analyze(self):
        """Measure texture detail, noise level, and shadow structure in skin."""
        raise NotImplementedError("SkinEngine.analyze is not implemented yet")

    def process(self):
        """Denoise compression artifacts while preserving pores and texture."""
        raise NotImplementedError("SkinEngine.process is not implemented yet")

    def export(self, output_dir=None):
        """Write the prepared skin layer and analysis data."""
        raise NotImplementedError("SkinEngine.export is not implemented yet")
