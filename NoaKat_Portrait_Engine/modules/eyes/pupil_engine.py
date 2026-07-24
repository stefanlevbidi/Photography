"""
Pupil Engine -- isolates the pupil and, where present, the catchlight.

The pupil is the darkest, highest-priority density point in a portrait
face; the catchlight (if any) is the brightest. Both anchor the eye's
perceived life and must survive tonal compression intact.

Not implemented yet -- scaffolding for the next development pass.
"""


class PupilEngine:
    """Analyzes and prepares the pupil/catchlight of a single eye."""

    def __init__(self, iris_crop):
        self.iris_crop = iris_crop
        self.pupil = {}

    def analyze(self):
        """Locate the pupil boundary and any catchlight highlight."""
        raise NotImplementedError("PupilEngine.analyze is not implemented yet")

    def process(self):
        """Preserve pupil darkness and catchlight brightness for density prep."""
        raise NotImplementedError("PupilEngine.process is not implemented yet")

    def export(self, output_dir=None):
        """Write pupil mask and analysis data."""
        raise NotImplementedError("PupilEngine.export is not implemented yet")
