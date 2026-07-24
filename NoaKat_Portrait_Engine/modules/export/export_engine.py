"""
Export Engine -- packages NoaKat's output for Astute Graphics Stipplism.

Writes final_density_map.png, inverted_density_map.png, per-person masks,
and metadata.json describing what was prepared. NoaKat's job ends here --
Stipplism takes this output and creates the dots.

Not implemented yet -- scaffolding for the next development pass.
"""

from config import settings


class ExportEngine:
    """Assembles and writes the final NoaKat output package."""

    def __init__(self, output_dir=None):
        self.output_dir = output_dir or settings.OUTPUT_DIR

    def analyze(self):
        """Verify that all required inputs (density maps, masks) are present."""
        raise NotImplementedError("ExportEngine.analyze is not implemented yet")

    def process(self):
        """Assemble the output package (density maps, masks, metadata)."""
        raise NotImplementedError("ExportEngine.process is not implemented yet")

    def export(self):
        """Write everything to output/ in the NoaKat directory layout."""
        raise NotImplementedError("ExportEngine.export is not implemented yet")
