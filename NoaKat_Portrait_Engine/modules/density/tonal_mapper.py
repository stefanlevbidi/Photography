"""
Tonal Mapper -- the curve behind the Density Engine's compression.

Maps the source tonal range onto a fixed number of density levels using an
equal-population histogram curve: bin boundaries are placed so each of the
`levels` output steps covers roughly the same number of pixels, rather than
an equal slice of the 0-255 range. In a portrait, most pixels sit in the
midtones (skin, hair mid-shadows), so this naturally spends more of the
level budget protecting midtone detail instead of wasting levels on rarely
used shadow/highlight extremes.

The pure-black background BackgroundEngine produces is excluded when
building the curve: it's a flat spike of literal 0s covering a large area,
not photographic content, and left in would consume most of the level
budget on "black" while compressing the actual subject into a handful of
levels. Excluding it does not affect how input value 0 itself gets mapped
(it still lands in the lowest bin, i.e. output 0) -- it only stops that
spike from distorting where the *other* boundaries fall.
"""

import json

import numpy as np


class TonalMapper:
    """Builds and applies a tone curve that quantizes grayscale into levels."""

    def __init__(self, levels):
        self.levels = levels
        self.histogram = None
        self.curve = None

    # ------------------------------------------------------------------
    def analyze(self, gray_image):
        """Inspect the source histogram to decide where the curve should bend."""
        histogram, _ = np.histogram(gray_image, bins=256, range=(0, 256))
        self.histogram = histogram
        return self.histogram

    # ------------------------------------------------------------------
    def process(self):
        """Build the quantization curve (a 256-entry lookup table) for self.levels steps."""
        if self.histogram is None:
            raise RuntimeError("TonalMapper.analyze must run before process")

        boundary_histogram = self.histogram.copy()
        boundary_histogram[0] = 0  # exclude the flat background spike, see module docstring

        total = int(boundary_histogram.sum())
        cdf = np.cumsum(boundary_histogram)

        if total == 0:
            self.curve = np.zeros(256, dtype=np.uint8)
            return self.curve

        boundaries = []
        for step in range(1, self.levels):
            target = total * step / self.levels
            boundary = int(np.searchsorted(cdf, target))
            boundaries.append(min(boundary, 255))

        curve = np.zeros(256, dtype=np.uint8)
        bin_index = np.searchsorted(boundaries, np.arange(256), side="right")
        if self.levels > 1:
            output_values = np.round(bin_index * 255 / (self.levels - 1)).astype(np.uint8)
        else:
            output_values = np.zeros(256, dtype=np.uint8)
        curve[:] = output_values

        self.curve = curve
        return self.curve

    # ------------------------------------------------------------------
    def export(self, output_path=None):
        """Write the tone curve definition for reuse/inspection."""
        if self.curve is None:
            self.process()

        data = {"levels": self.levels, "curve": self.curve.tolist()}
        if output_path is not None:
            with open(output_path, "w") as f:
                json.dump(data, f, indent=2)
        return data
