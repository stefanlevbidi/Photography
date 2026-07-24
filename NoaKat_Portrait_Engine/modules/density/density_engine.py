"""
Density Engine -- the heart of NoaKat.

Compresses full 256-level grayscale information into a controlled band of
25-40 meaningful tonal density levels, preserving the detail Astute Graphics
Stipplism needs to place dots -- without flattening the image and without
generating any dots, points, or artwork itself.

Uses TonalMapper's equal-population curve so the level budget is spent
where the image actually has detail, instead of uniformly across the full
0-255 range. Produces both the density map and its tonal inverse, since
different Stipplism setups expect density encoded either as "dark = dense"
or "light = dense".
"""

from pathlib import Path

import cv2

from config import settings
from modules.density.tonal_mapper import TonalMapper


class DensityEngine:
    """Analyzes brightness distribution and compresses it into density levels."""

    def __init__(self, image_path, levels=settings.DENSITY_LEVELS_DEFAULT):
        self.image_path = Path(image_path)
        self.levels = levels
        self.image = None
        self.gray = None
        self.tonal_mapper = None
        self.density_map = None
        self.inverted_density_map = None

    # ------------------------------------------------------------------
    def analyze(self):
        """Analyze brightness distribution and identify detail-critical regions."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")
        self.image = image
        self.gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        self.tonal_mapper = TonalMapper(self.levels)
        histogram = self.tonal_mapper.analyze(self.gray)
        return {"histogram": histogram, "levels": self.levels}

    # ------------------------------------------------------------------
    def process(self):
        """Compress tones into self.levels density steps, preserving detail."""
        if self.tonal_mapper is None:
            self.analyze()

        curve = self.tonal_mapper.process()
        self.density_map = cv2.LUT(self.gray, curve)
        self.inverted_density_map = 255 - self.density_map
        return self.density_map, self.inverted_density_map

    # ------------------------------------------------------------------
    def export(self, output_dir=None, person_id="person_01"):
        """Write the density map and its inverse for Stipplism import."""
        if self.density_map is None:
            self.process()

        output_dir = Path(output_dir) if output_dir else settings.MASKS_DIR
        person_dir = output_dir / person_id / "density"
        person_dir.mkdir(parents=True, exist_ok=True)

        density_path = person_dir / settings.FINAL_DENSITY_MAP_NAME
        inverted_path = person_dir / settings.INVERTED_DENSITY_MAP_NAME
        curve_path = person_dir / "tonal_curve.json"

        cv2.imwrite(str(density_path), self.density_map)
        cv2.imwrite(str(inverted_path), self.inverted_density_map)
        self.tonal_mapper.export(curve_path)

        return [density_path, inverted_path, curve_path]
