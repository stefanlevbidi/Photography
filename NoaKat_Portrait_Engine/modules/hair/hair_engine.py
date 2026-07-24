"""
Hair Engine -- preserves hair mass, highlights, texture, and direction.

Hair is a major source of realistic density variation in a stippled
portrait: dark masses read as high density, highlights read as low
density, and strand direction should stay visible in the flow of tone.

Not implemented yet -- scaffolding for the next development pass.
"""


class HairEngine:
    """Analyzes and prepares the hair region for density mapping."""

    def __init__(self, image_path, hair_mask):
        self.image_path = image_path
        self.hair_mask = hair_mask

    def analyze(self):
        """Measure dark mass distribution, highlights, and strand direction."""
        raise NotImplementedError("HairEngine.analyze is not implemented yet")

    def process(self):
        """Prepare hair tonal detail preserving depth and directional texture."""
        raise NotImplementedError("HairEngine.process is not implemented yet")

    def export(self, output_dir=None):
        """Write the prepared hair layer and analysis data."""
        raise NotImplementedError("HairEngine.export is not implemented yet")
