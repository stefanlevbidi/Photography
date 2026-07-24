"""
Image Analyzer -- the entry point of the NoaKat Portrait Engine pipeline.

Responsibility: understand the raw photo before anything else touches it.
It never modifies pixels; it only measures them and reports findings that
downstream engines (face, skin, hair, background, density) rely on.

    PHOTO -> [ANALYSIS] -> intelligent processing -> density prep -> export
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image

from config import settings


class ImageAnalyzer:
    """Analyzes a portrait photo and reports its technical characteristics."""

    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.image = None
        self.analysis = {}

    # ------------------------------------------------------------------
    def analyze(self):
        """Load the image and run every measurement pass."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        self.image = Image.open(self.image_path)
        # Force a decode now so truncated/corrupt files fail here, not later.
        self.image.load()

        rgb_image = self.image.convert("RGB")
        gray_array = np.asarray(rgb_image.convert("L"), dtype=np.float64)

        self.analysis = {
            "file": self._analyze_file(),
            "resolution": self._analyze_resolution(),
            "color_mode": self._analyze_color_mode(),
            "brightness": self._analyze_brightness(gray_array),
            "contrast": self._analyze_contrast(gray_array),
            "histogram": self._analyze_histogram(gray_array),
            "exposure_problems": self._analyze_exposure(gray_array),
        }
        return self.analysis

    # ------------------------------------------------------------------
    def process(self):
        """Derive higher-level flags from the raw measurements."""
        if not self.analysis:
            self.analyze()

        problems = list(self.analysis["exposure_problems"]["problems"])
        min_w, min_h = settings.MIN_RECOMMENDED_RESOLUTION
        width, height = self.analysis["resolution"]["width"], self.analysis["resolution"]["height"]
        if width < min_w or height < min_h:
            problems.append("resolution_below_recommended_minimum")

        self.analysis["exposure_problems"]["problems"] = problems
        self.analysis["is_ready_for_pipeline"] = len(problems) == 0
        return self.analysis

    # ------------------------------------------------------------------
    def export(self, output_path=None):
        """Write the analysis to JSON and return the path written to."""
        if not self.analysis:
            self.process()

        output_path = Path(output_path) if output_path else settings.OUTPUT_DIR / "analysis.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.analysis, f, indent=2)
        return output_path

    # ------------------------------------------------------------------
    # Individual measurement passes
    # ------------------------------------------------------------------
    def _analyze_file(self):
        return {
            "path": str(self.image_path),
            "name": self.image_path.name,
            "format": self.image.format,
            "size_bytes": self.image_path.stat().st_size,
        }

    def _analyze_resolution(self):
        width, height = self.image.size
        return {
            "width": width,
            "height": height,
            "megapixels": round((width * height) / 1_000_000, 2),
        }

    def _analyze_color_mode(self):
        return {
            "mode": self.image.mode,
            "has_alpha": self.image.mode in ("RGBA", "LA", "PA"),
            "is_grayscale": self.image.mode in ("L", "LA", "1"),
        }

    def _analyze_brightness(self, gray_array):
        mean_brightness = float(np.mean(gray_array))
        return {
            "mean": round(mean_brightness, 2),
            "min": float(np.min(gray_array)),
            "max": float(np.max(gray_array)),
        }

    def _analyze_contrast(self, gray_array):
        std_dev = float(np.std(gray_array))
        return {
            "std_dev": round(std_dev, 2),
            "is_low_contrast": std_dev < settings.CONTRAST_LOW_THRESHOLD,
        }

    def _analyze_histogram(self, gray_array):
        histogram, _ = np.histogram(gray_array, bins=256, range=(0, 256))
        return {
            "bins": 256,
            "counts": histogram.tolist(),
            "shadow_ratio": round(float(np.sum(histogram[:16])) / gray_array.size, 4),
            "highlight_ratio": round(float(np.sum(histogram[240:])) / gray_array.size, 4),
        }

    def _analyze_exposure(self, gray_array):
        mean_brightness = float(np.mean(gray_array))
        total_pixels = gray_array.size

        clipped_shadows = float(np.sum(gray_array <= 2)) / total_pixels
        clipped_highlights = float(np.sum(gray_array >= 253)) / total_pixels

        problems = []
        if mean_brightness < settings.BRIGHTNESS_LOW_THRESHOLD:
            problems.append("underexposed")
        if mean_brightness > settings.BRIGHTNESS_HIGH_THRESHOLD:
            problems.append("overexposed")
        if clipped_shadows > settings.CLIPPED_SHADOW_RATIO:
            problems.append("clipped_shadows")
        if clipped_highlights > settings.CLIPPED_HIGHLIGHT_RATIO:
            problems.append("clipped_highlights")
        if np.std(gray_array) < settings.CONTRAST_LOW_THRESHOLD:
            problems.append("flat_low_contrast")

        return {
            "clipped_shadow_ratio": round(clipped_shadows, 4),
            "clipped_highlight_ratio": round(clipped_highlights, 4),
            "problems": problems,
        }
