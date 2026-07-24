"""
Tonal Mapper -- the curve behind the Density Engine's compression.

Maps the source tonal range onto a fixed number of density levels using a
curve that protects midtone detail (skin, eyes) and shadow/highlight
separation (hair, background), rather than a plain linear quantization.

Not implemented yet -- scaffolding for the next development pass.
"""


class TonalMapper:
    """Builds and applies a tone curve that quantizes grayscale into levels."""

    def __init__(self, levels):
        self.levels = levels
        self.curve = None

    def analyze(self):
        """Inspect the source histogram to decide where the curve should bend."""
        raise NotImplementedError("TonalMapper.analyze is not implemented yet")

    def process(self):
        """Build the quantization curve for self.levels density steps."""
        raise NotImplementedError("TonalMapper.process is not implemented yet")

    def export(self, output_path=None):
        """Write the tone curve definition for reuse/inspection."""
        raise NotImplementedError("TonalMapper.export is not implemented yet")
