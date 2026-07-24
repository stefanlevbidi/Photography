"""
Iris Engine -- isolates iris detail within a detected eye.

Preserves natural iris texture and tonal variation so the density engine
has real information to compress, instead of a flat disc.

Not implemented yet -- scaffolding for the next development pass.
"""


class IrisEngine:
    """Analyzes and prepares the iris region of a single eye."""

    def __init__(self, eye_crop):
        self.eye_crop = eye_crop
        self.iris = {}

    def analyze(self):
        """Locate the iris boundary and texture within the eye crop."""
        raise NotImplementedError("IrisEngine.analyze is not implemented yet")

    def process(self):
        """Preserve iris texture and tonal variation for density prep."""
        raise NotImplementedError("IrisEngine.process is not implemented yet")

    def export(self, output_dir=None):
        """Write iris mask and analysis data."""
        raise NotImplementedError("IrisEngine.export is not implemented yet")
