"""
Skin Engine -- prepares skin for stippling without faking it.

Works only within the skin mask produced by FaceRegions. Uses frequency
separation on the luminance channel: a bilateral-filtered "tone" layer
(shadows, skin color transitions) is kept untouched, while the leftover
"detail" layer (pores, fine texture, sensor noise, compression blockiness)
gets a very mild median filter -- enough to knock down random noise
speckle without erasing the structured texture that stippling needs.
Color channels are left untouched so skin tone doesn't shift.

Everything outside the skin mask (eyes, lips, hair, background, etc.)
passes through unmodified -- this engine only ever touches skin pixels.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from config import settings


class SkinEngine:
    """Analyzes and prepares the skin region for density mapping."""

    def __init__(self, image_path, skin_mask):
        self.image_path = Path(image_path)
        self.skin_mask = skin_mask
        self.image = None
        self.noise_estimate = None
        self.prepared_image = None

    # ------------------------------------------------------------------
    def analyze(self):
        """Measure noise level in the skin region before any processing."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")
        self.image = image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        skin_pixels = gray[self.skin_mask > 0]
        if skin_pixels.size == 0:
            self.noise_estimate = 0.0
            return {"noise_estimate": self.noise_estimate}

        median = cv2.medianBlur(gray, 3)
        residual = cv2.absdiff(gray, median)[self.skin_mask > 0]
        self.noise_estimate = float(np.mean(residual))
        return {"noise_estimate": self.noise_estimate}

    # ------------------------------------------------------------------
    def process(self):
        """Denoise skin pixels via frequency separation, preserving texture."""
        if self.image is None:
            self.analyze()

        if (self.skin_mask > 0).sum() == 0:
            self.prepared_image = self.image.copy()
            return self.prepared_image

        lab = cv2.cvtColor(self.image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        tone = cv2.bilateralFilter(l_channel, d=9, sigmaColor=50, sigmaSpace=50)
        detail = l_channel.astype(np.int16) - tone.astype(np.int16)

        detail_shifted = np.clip(detail + 128, 0, 255).astype(np.uint8)
        detail_denoised = cv2.medianBlur(detail_shifted, 3).astype(np.int16) - 128

        prepared_l = np.clip(tone.astype(np.int16) + detail_denoised, 0, 255).astype(np.uint8)
        prepared_lab = cv2.merge([prepared_l, a_channel, b_channel])
        prepared_bgr = cv2.cvtColor(prepared_lab, cv2.COLOR_LAB2BGR)

        mask_3ch = cv2.cvtColor(self.skin_mask, cv2.COLOR_GRAY2BGR)
        self.prepared_image = np.where(mask_3ch > 0, prepared_bgr, self.image)
        return self.prepared_image

    # ------------------------------------------------------------------
    def export(self, output_dir=None, person_id="person_01"):
        """Write the prepared skin image and before/after noise metrics."""
        if self.prepared_image is None:
            self.process()

        output_dir = Path(output_dir) if output_dir else settings.MASKS_DIR
        person_dir = output_dir / person_id / "skin"
        person_dir.mkdir(parents=True, exist_ok=True)

        prepared_path = person_dir / "prepared.png"
        cv2.imwrite(str(prepared_path), self.prepared_image)

        gray_after = cv2.cvtColor(self.prepared_image, cv2.COLOR_BGR2GRAY)
        median_after = cv2.medianBlur(gray_after, 3)
        skin_pixels_after = gray_after[self.skin_mask > 0]
        if skin_pixels_after.size > 0:
            residual_after = cv2.absdiff(gray_after, median_after)[self.skin_mask > 0]
            noise_after = float(np.mean(residual_after))
        else:
            noise_after = 0.0

        analysis_path = person_dir / "analysis.json"
        with open(analysis_path, "w") as f:
            json.dump(
                {
                    "person_id": person_id,
                    "noise_estimate_before": self.noise_estimate,
                    "noise_estimate_after": noise_after,
                },
                f, indent=2,
            )

        return [prepared_path, analysis_path]
