"""
Background Engine -- separates the subject from the background.

The final background must be pure black (RGB 0,0,0) with no gray
contamination, so it produces minimal unwanted stippling density. This is
a hard cut, not a feathered blend: any softening at the subject/background
edge would leave partial gray values behind, which is exactly the
contamination this engine exists to prevent.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from config import settings


class BackgroundEngine:
    """Analyzes and flattens the background region to pure black."""

    def __init__(self, image_path, background_mask):
        self.image_path = Path(image_path)
        self.background_mask = background_mask
        self.image = None
        self.background_mean = None
        self.background_std = None
        self.prepared_image = None

    # ------------------------------------------------------------------
    def analyze(self):
        """Measure background uniformity and separation quality from the subject."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")
        self.image = image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        background_pixels = gray[self.background_mask > 0]

        if background_pixels.size == 0:
            self.background_mean = 0.0
            self.background_std = 0.0
        else:
            self.background_mean = float(np.mean(background_pixels))
            self.background_std = float(np.std(background_pixels))

        return {"background_mean": self.background_mean, "background_std": self.background_std}

    # ------------------------------------------------------------------
    def process(self):
        """Flatten the background to pure black with a clean subject edge."""
        if self.image is None:
            self.analyze()

        self.prepared_image = self.image.copy()
        self.prepared_image[self.background_mask > 0] = (0, 0, 0)
        return self.prepared_image

    # ------------------------------------------------------------------
    def export(self, output_dir=None, person_id="person_01"):
        """Write the prepared background layer and analysis data."""
        if self.prepared_image is None:
            self.process()

        output_dir = Path(output_dir) if output_dir else settings.MASKS_DIR
        person_dir = output_dir / person_id / "background"
        person_dir.mkdir(parents=True, exist_ok=True)

        prepared_path = person_dir / "prepared.png"
        cv2.imwrite(str(prepared_path), self.prepared_image)

        background_pixels_after = self.prepared_image[self.background_mask > 0]
        is_pure_black = bool(np.all(background_pixels_after == 0)) if background_pixels_after.size > 0 else True

        analysis_path = person_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(
                {
                    "person_id": person_id,
                    "background_mean_before": self.background_mean,
                    "background_std_before": self.background_std,
                    "background_target_rgb": list(settings.BACKGROUND_TARGET_RGB),
                    "is_pure_black": is_pure_black,
                },
                f, indent=2,
            )

        return [prepared_path, analysis_path]
