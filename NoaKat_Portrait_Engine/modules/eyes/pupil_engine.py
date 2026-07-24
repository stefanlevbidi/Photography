"""
Pupil Engine -- isolates the pupil and, where present, the catchlight.

The pupil is the darkest, highest-priority density point in a portrait
face; the catchlight (if any) is the brightest. Both anchor the eye's
perceived life and must survive tonal compression intact. Both are found
by thresholding within the iris region already located by IrisEngine --
the pupil as the darkest connected blob, the catchlight as a small,
very bright connected blob -- rather than assumed or drawn from scratch.
"""

import cv2
import numpy as np


class PupilEngine:
    """Analyzes and prepares the pupil/catchlight within a located iris."""

    def __init__(self, eye_crop_gray, iris):
        self.eye_crop_gray = eye_crop_gray
        self.iris = iris
        self.pupil = None
        self.catchlight = None
        self.pupil_mask = None
        self.catchlight_mask = None

    # ------------------------------------------------------------------
    def analyze(self):
        """Locate the pupil boundary and any catchlight highlight."""
        height, width = self.eye_crop_gray.shape[:2]
        iris_mask = np.zeros((height, width), dtype=np.uint8)
        center = (int(round(self.iris["cx"])), int(round(self.iris["cy"])))
        radius = max(1, int(round(self.iris["r"])))
        cv2.circle(iris_mask, center, radius, 255, -1)

        iris_pixels = self.eye_crop_gray[iris_mask > 0]
        if iris_pixels.size == 0:
            self.pupil = {"cx": float(center[0]), "cy": float(center[1]), "r": float(radius * 0.4)}
            self.catchlight = None
            return {"pupil": self.pupil, "catchlight": self.catchlight}

        # Pupil: darkest cluster within the iris, split via Otsu thresholding
        # (adapts to the actual histogram instead of assuming a fixed
        # percentile, which breaks down when the pupil is a small fraction
        # of the iris area).
        otsu_threshold, _ = cv2.threshold(
            iris_pixels.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        dark_mask = np.where((self.eye_crop_gray <= otsu_threshold) & (iris_mask > 0), 255, 0).astype(np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        pupil_candidate = self._largest_blob_as_circle(dark_mask, fallback_radius=radius * 0.4, center=center)

        # If the "pupil" comes back nearly as large as the iris itself, the
        # eye lacked enough contrast for a real split (e.g. low light, small
        # crop) -- fall back to a documented anatomical proportion instead
        # of reporting a pupil that isn't actually a pupil.
        if pupil_candidate["r"] > radius * 0.85:
            pupil_candidate = {"cx": float(center[0]), "cy": float(center[1]), "r": float(radius * 0.4)}
        self.pupil = pupil_candidate

        # Catchlight: a small, very bright blob (camera/light reflection).
        bright_threshold = max(200, np.percentile(iris_pixels, 97))
        bright_mask = np.where((self.eye_crop_gray >= bright_threshold) & (iris_mask > 0), 255, 0).astype(np.uint8)
        max_catchlight_area = np.pi * (radius ** 2) * 0.3
        self.catchlight = self._largest_blob_as_circle(
            bright_mask, fallback_radius=None, center=None, max_area=max_catchlight_area
        )

        return {"pupil": self.pupil, "catchlight": self.catchlight}

    def _largest_blob_as_circle(self, binary_mask, fallback_radius, center, max_area=None):
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            if fallback_radius is None:
                return None
            return {"cx": float(center[0]), "cy": float(center[1]), "r": float(fallback_radius)}

        candidates = contours
        if max_area is not None:
            candidates = [c for c in contours if cv2.contourArea(c) <= max_area]
            if not candidates:
                return None

        largest = max(candidates, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(largest)
        if r < 1:
            if fallback_radius is None:
                return None
            return {"cx": float(center[0]), "cy": float(center[1]), "r": float(fallback_radius)}
        return {"cx": float(cx), "cy": float(cy), "r": float(r)}

    # ------------------------------------------------------------------
    def process(self):
        """Rasterize pupil and catchlight masks sized to the eye crop."""
        if self.pupil is None:
            self.analyze()

        height, width = self.eye_crop_gray.shape[:2]
        pupil_mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(
            pupil_mask,
            (int(round(self.pupil["cx"])), int(round(self.pupil["cy"]))),
            max(1, int(round(self.pupil["r"]))),
            255, -1,
        )
        self.pupil_mask = pupil_mask

        catchlight_mask = np.zeros((height, width), dtype=np.uint8)
        if self.catchlight is not None:
            cv2.circle(
                catchlight_mask,
                (int(round(self.catchlight["cx"])), int(round(self.catchlight["cy"]))),
                max(1, int(round(self.catchlight["r"]))),
                255, -1,
            )
        self.catchlight_mask = catchlight_mask

        return {"pupil_mask": pupil_mask, "catchlight_mask": catchlight_mask}

    # ------------------------------------------------------------------
    def export(self):
        """Return the pupil/catchlight analysis and masks for the caller to persist."""
        if self.pupil_mask is None:
            self.process()
        return {
            "pupil": self.pupil,
            "catchlight": self.catchlight,
            "pupil_mask": self.pupil_mask,
            "catchlight_mask": self.catchlight_mask,
        }
