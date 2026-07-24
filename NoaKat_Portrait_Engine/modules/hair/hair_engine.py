"""
Hair Engine -- preserves hair mass, highlights, texture, and direction.

Hair is a major source of realistic density variation in a stippled
portrait: dark masses read as high density, highlights read as low
density, and strand direction should stay visible in the flow of tone.

Unlike SkinEngine (which removes noise), this engine enhances local
contrast within the hair region via CLAHE on the luminance channel --
pulling dark masses and highlights further apart so stippling has real
tonal range to work with, instead of flattening hair into a mid-gray
blob. Strand direction is measured (not drawn) from the dominant gradient
orientation, and reported as metadata for later stages.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from config import settings


class HairEngine:
    """Analyzes and prepares the hair region for density mapping."""

    def __init__(self, image_path, hair_mask):
        self.image_path = Path(image_path)
        self.hair_mask = hair_mask
        self.image = None
        self.dark_mass_ratio = None
        self.highlight_ratio = None
        self.dominant_angle_degrees = None
        self.prepared_image = None

    # ------------------------------------------------------------------
    def analyze(self):
        """Measure dark mass distribution, highlights, and strand direction."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")
        self.image = image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hair_pixels = gray[self.hair_mask > 0]

        if hair_pixels.size == 0:
            self.dark_mass_ratio = 0.0
            self.highlight_ratio = 0.0
            self.dominant_angle_degrees = None
            return self._analysis_summary()

        dark_threshold, _ = cv2.threshold(
            hair_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.dark_mass_ratio = float(np.mean(hair_pixels <= dark_threshold))
        self.highlight_ratio = float(np.mean(hair_pixels >= 200))

        self.dominant_angle_degrees = self._estimate_dominant_direction(gray)
        return self._analysis_summary()

    def _estimate_dominant_direction(self, gray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gx, gy)

        mask_bool = self.hair_mask > 0
        if not np.any(mask_bool) or np.sum(magnitude[mask_bool]) == 0:
            return None

        # Strand direction runs perpendicular to the local gradient; average
        # via double-angle (mod-pi) so opposite-facing edges don't cancel out.
        angles = np.arctan2(gy[mask_bool], gx[mask_bool])
        weights = magnitude[mask_bool]
        mean_sin = np.sum(weights * np.sin(2 * angles))
        mean_cos = np.sum(weights * np.cos(2 * angles))
        gradient_angle = 0.5 * np.arctan2(mean_sin, mean_cos)
        strand_angle = gradient_angle + np.pi / 2
        return float(np.degrees(strand_angle) % 180)

    def _analysis_summary(self):
        return {
            "dark_mass_ratio": self.dark_mass_ratio,
            "highlight_ratio": self.highlight_ratio,
            "dominant_angle_degrees": self.dominant_angle_degrees,
        }

    # ------------------------------------------------------------------
    def process(self):
        """Prepare hair tonal detail preserving depth and directional texture."""
        if self.image is None:
            self.analyze()

        if (self.hair_mask > 0).sum() == 0:
            self.prepared_image = self.image.copy()
            return self.prepared_image

        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced_l = clahe.apply(l_channel)

        prepared_lab = cv2.merge([enhanced_l, a_channel, b_channel])
        prepared_bgr = cv2.cvtColor(prepared_lab, cv2.COLOR_LAB2BGR)

        mask_3ch = cv2.cvtColor(self.hair_mask, cv2.COLOR_GRAY2BGR)
        self.prepared_image = np.where(mask_3ch > 0, prepared_bgr, self.image)
        return self.prepared_image

    # ------------------------------------------------------------------
    def export(self, output_dir=None, person_id="person_01"):
        """Write the prepared hair layer and analysis data."""
        if self.prepared_image is None:
            self.process()

        output_dir = Path(output_dir) if output_dir else settings.MASKS_DIR
        person_dir = output_dir / person_id / "hair"
        person_dir.mkdir(parents=True, exist_ok=True)

        prepared_path = person_dir / "prepared.png"
        cv2.imwrite(str(prepared_path), self.prepared_image)

        analysis_path = person_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump({"person_id": person_id, **self._analysis_summary()}, f, indent=2)

        return [prepared_path, analysis_path]
