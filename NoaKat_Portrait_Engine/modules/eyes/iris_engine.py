"""
Iris Engine -- isolates iris detail within a detected eye.

Preserves natural iris texture and tonal variation so the density engine
has real information to compress, instead of a flat disc. The iris circle
is located with a Hough circle transform on the eye crop -- a real
detection, not a guess -- falling back to a proportional estimate only
when no circle is confidently found (e.g. a closed or heavily shadowed
eye).
"""

import cv2
import numpy as np


class IrisEngine:
    """Analyzes and prepares the iris region of a single eye crop."""

    def __init__(self, eye_crop_gray):
        self.eye_crop_gray = eye_crop_gray
        self.iris = None
        self.mask = None

    # ------------------------------------------------------------------
    def analyze(self):
        """Locate the iris circle (center, radius) within the eye crop."""
        height, width = self.eye_crop_gray.shape[:2]
        blurred = cv2.GaussianBlur(self.eye_crop_gray, (5, 5), 0)

        min_dim = min(height, width)
        circles = cv2.HoughCircles(
            blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=min_dim,
            param1=50, param2=15,
            minRadius=int(min_dim * 0.2), maxRadius=int(min_dim * 0.48),
        )

        if circles is not None:
            cx, cy, r = circles[0][0]
        else:
            # No confident circle (closed/shadowed eye) -- fall back to a
            # centered disc proportional to the crop, a documented estimate.
            cx, cy, r = width / 2, height / 2, min_dim * 0.35

        self.iris = {"cx": float(cx), "cy": float(cy), "r": float(r)}
        return self.iris

    # ------------------------------------------------------------------
    def process(self):
        """Rasterize the iris circle into a mask sized to the eye crop."""
        if self.iris is None:
            self.analyze()

        height, width = self.eye_crop_gray.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        center = (int(round(self.iris["cx"])), int(round(self.iris["cy"])))
        radius = int(round(self.iris["r"]))
        cv2.circle(mask, center, radius, 255, -1)

        self.mask = mask
        return mask

    # ------------------------------------------------------------------
    def export(self):
        """Return the iris analysis and mask for the caller to persist."""
        if self.mask is None:
            self.process()
        return {"iris": self.iris, "mask": self.mask}
