"""
Face Regions -- builds accurate masks for one detected person.

Produces the masks every other engine consumes: face, skin, eyes,
eyebrows, lips, hair, clothing, background. Masks are per-person so
group portraits keep each subject's regions independent.

Region detail, by how it's derived:
- eyes: detected directly with OpenCV's Haar eye cascade, restricted to
  the face box from FaceDetector (reliable, not a guess).
- foreground/background: segmented with OpenCV's GrabCut, seeded from the
  face box expanded to cover hair and torso (a real segmentation, not a
  guess) -- this is what makes the background mask usable for the
  Background Engine.
- eyebrows and lips: no landmark model is available in this environment
  (mediapipe's legacy Face Mesh API and downloadable dlib/YuNet models
  are not reachable here), so these two are placed using standard facial
  proportions relative to the detected eyes and face box rather than
  detected directly. They're good enough to seed density preparation, but
  are the one part of this module that's a heuristic, not a detection.
- face/skin/hair/clothing: derived by combining the above with the face
  box geometry.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from config import settings

EYE_CASCADE_PATH = Path(__file__).resolve().parent / "haarcascade_eye.xml"


class FaceRegions:
    """Builds segmentation masks for a single detected person."""

    REGION_NAMES = (
        "face", "skin", "eyes", "eyebrows", "lips", "hair", "clothing", "background",
    )

    def __init__(self, image_path, face_box, eye_cascade_path=EYE_CASCADE_PATH):
        self.image_path = Path(image_path)
        self.face_box = face_box
        self.eye_cascade_path = Path(eye_cascade_path)
        self.image = None
        self.image_size = None
        self.eyes = []
        self.foreground_mask = None
        self.masks = {}

    # ------------------------------------------------------------------
    def analyze(self):
        """Detect eyes within the face box and segment person from background."""
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        image = cv2.imread(str(self.image_path))
        if image is None:
            raise ValueError(f"Could not read image: {self.image_path}")

        self.image = image
        height, width = image.shape[:2]
        self.image_size = (width, height)

        self.eyes = self._detect_eyes(image)
        self.foreground_mask = self._segment_foreground(image)
        return {"eyes": self.eyes, "foreground_mask": self.foreground_mask}

    def _detect_eyes(self, image):
        fb = self.face_box
        pad_x, pad_y = int(fb["width"] * 0.1), int(fb["height"] * 0.1)
        rx0 = max(0, fb["x"] - pad_x)
        ry0 = max(0, fb["y"] - pad_y)
        rx1 = min(image.shape[1], fb["x"] + fb["width"] + pad_x)
        ry1 = min(image.shape[0], fb["y"] + fb["height"] + pad_y)

        roi_gray = cv2.cvtColor(image[ry0:ry1, rx0:rx1], cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier(str(self.eye_cascade_path))
        if cascade.empty():
            raise RuntimeError(f"Could not load cascade classifier: {self.eye_cascade_path}")

        detections = cascade.detectMultiScale(
            roi_gray, scaleFactor=1.05, minNeighbors=8, minSize=(20, 20)
        )
        # Only the upper two-thirds of the face box can plausibly contain eyes.
        upper_limit = ry0 + int((ry1 - ry0) * 0.66)
        candidates = [
            {"x": int(x + rx0), "y": int(y + ry0), "width": int(w), "height": int(h)}
            for (x, y, w, h) in detections
            if (y + ry0 + h) <= upper_limit
        ]
        # Keep the two largest (most confident) detections, ordered left-to-right.
        candidates.sort(key=lambda e: e["width"] * e["height"], reverse=True)
        top_two = candidates[:2]
        top_two.sort(key=lambda e: e["x"])
        return top_two

    def _segment_foreground(self, image):
        fb = self.face_box
        height, width = image.shape[:2]
        # A tighter seed rect keeps GrabCut from wandering into unrelated
        # background structure (e.g. furniture, frames) far from the face --
        # wide padding was pulling in disconnected background as "foreground".
        x0 = max(0, int(fb["x"] - fb["width"] * 0.4))
        y0 = max(0, int(fb["y"] - fb["height"] * 0.5))
        x1 = min(width, int(fb["x"] + fb["width"] * 1.4))
        y1 = min(height, int(fb["y"] + fb["height"] * 4.0))
        rect = (x0, y0, x1 - x0, y1 - y0)

        mask = np.zeros((height, width), dtype=np.uint8)
        bgd_model = np.zeros((1, 65), dtype=np.float64)
        fgd_model = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 8, cv2.GC_INIT_WITH_RECT)

        # A second pass hard-forces regions GrabCut's colour model tends to
        # misjudge: dark/voluminous hair and dark clothing both read as
        # near-identical to a black background by colour alone, so a
        # colour-only first pass crops hair short and can erase dark
        # garments. Hard priors (GC_FGD) are never revisited by GrabCut, so
        # they survive even when local colour statistics point elsewhere.
        cx = fb["x"] + fb["width"] // 2
        face_top, face_bottom = fb["y"], fb["y"] + fb["height"]

        # Hair: a full ellipse (not just the cap above the head) reaching
        # down into the face box, so the forced region stays contiguous
        # with the reliably-detected face/skin blob -- a half-dome that
        # stops right at the hairline leaves a gap that GrabCut's second
        # pass can reclassify as background, disconnecting the hair mass
        # from the person and losing it to the largest-component filter.
        hair_center = (cx, face_top + int(fb["height"] * 0.15))
        hair_axes = (int(fb["width"] * 0.72), int(fb["height"] * 0.65))
        hair_prior = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(hair_prior, hair_center, hair_axes, 0, 0, 360, 255, -1)
        mask[hair_prior == 255] = cv2.GC_FGD

        # Torso/shoulders: clothing below the face, hard-forced foreground
        # so dark or patterned garments aren't merged into the background.
        torso_y0 = face_bottom - int(fb["height"] * 0.1)
        torso_y1 = min(height, face_bottom + int(fb["height"] * 2.0))
        torso_x0 = max(0, cx - int(fb["width"] * 1.0))
        torso_x1 = min(width, cx + int(fb["width"] * 1.0))
        mask[torso_y0:torso_y1, torso_x0:torso_x1] = cv2.GC_FGD

        bgd_model2 = np.zeros((1, 65), dtype=np.float64)
        fgd_model2 = np.zeros((1, 65), dtype=np.float64)
        cv2.grabCut(image, mask, None, bgd_model2, fgd_model2, 5, cv2.GC_INIT_WITH_MASK)

        foreground = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

        # Clean up thin spurious bridges to unrelated background, then keep
        # only the main connected mass (the person), discarding any leftover
        # disconnected fragments.
        kernel = np.ones((7, 7), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            foreground = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        # The hard hair/torso priors are geometric approximations, so they
        # inevitably sweep in some genuine background alongside the real
        # subject. Reclassify flat, near-black patches back to background:
        # a flat backdrop has near-zero local variance, while real dark hair
        # and clothing keep visible grain and texture even at their darkest.
        # Without this, those swept-in patches stay classified as hair or
        # clothing, so later engines process (and brighten) them instead of
        # flattening them to pure black -- leaving visible grey artifacts.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        local_mean = cv2.blur(gray, (9, 9))
        local_sqmean = cv2.blur(gray * gray, (9, 9))
        local_variance = np.clip(local_sqmean - local_mean * local_mean, 0, None)
        flat_black = (local_variance < 4) & (gray < 10)
        foreground[flat_black] = 0

        # Closing repairs the small holes this punches into genuinely
        # dark-but-flat subject pixels (a few isolated dark threads etc.)
        # without undoing the reclassification of real background patches.
        close_kernel = np.ones((5, 5), np.uint8)
        foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, close_kernel)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(foreground)
        if num_labels > 1:
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            foreground = np.where(labels == largest_label, 255, 0).astype(np.uint8)

        return foreground

    # ------------------------------------------------------------------
    def process(self):
        """Rasterize face, skin, eyes, eyebrows, lips, hair, clothing, background."""
        if self.foreground_mask is None:
            self.analyze()

        width, height = self.image_size
        fb = self.face_box
        blank = np.zeros((height, width), dtype=np.uint8)

        face_mask = blank.copy()
        center = (fb["x"] + fb["width"] // 2, fb["y"] + fb["height"] // 2)
        axes = (fb["width"] // 2, int(fb["height"] * 0.6))
        cv2.ellipse(face_mask, center, axes, 0, 0, 360, 255, -1)

        eyes_mask = blank.copy()
        eyebrows_mask = blank.copy()
        for eye in self.eyes:
            cv2.rectangle(
                eyes_mask,
                (eye["x"], eye["y"]),
                (eye["x"] + eye["width"], eye["y"] + eye["height"]),
                255, -1,
            )
            brow_height = int(eye["height"] * 0.4)
            brow_y0 = max(0, eye["y"] - brow_height)
            cv2.rectangle(
                eyebrows_mask,
                (eye["x"], brow_y0),
                (eye["x"] + eye["width"], eye["y"]),
                255, -1,
            )

        lips_mask = blank.copy()
        lips_w = int(fb["width"] * 0.5)
        lips_h = int(fb["height"] * 0.15)
        lips_x = fb["x"] + (fb["width"] - lips_w) // 2
        lips_y = fb["y"] + int(fb["height"] * 0.72)
        cv2.rectangle(lips_mask, (lips_x, lips_y), (lips_x + lips_w, lips_y + lips_h), 255, -1)

        skin_mask = cv2.bitwise_and(face_mask, self.foreground_mask)
        skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(eyes_mask))
        skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(eyebrows_mask))
        skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(lips_mask))

        hair_mask = cv2.bitwise_and(self.foreground_mask, cv2.bitwise_not(face_mask))
        below_face = blank.copy()
        cv2.rectangle(below_face, (0, fb["y"] + fb["height"]), (width, height), 255, -1)
        clothing_mask = cv2.bitwise_and(hair_mask, below_face)
        hair_mask = cv2.bitwise_and(hair_mask, cv2.bitwise_not(below_face))

        background_mask = cv2.bitwise_not(self.foreground_mask)

        self.masks = {
            "face": face_mask,
            "skin": skin_mask,
            "eyes": eyes_mask,
            "eyebrows": eyebrows_mask,
            "lips": lips_mask,
            "hair": hair_mask,
            "clothing": clothing_mask,
            "background": background_mask,
        }
        return self.masks

    # ------------------------------------------------------------------
    def export(self, output_dir=None, person_id="person_01"):
        """Write each region mask as a PNG under output/masks/<person_id>/."""
        if not self.masks:
            self.process()

        output_dir = Path(output_dir) if output_dir else settings.MASKS_DIR
        person_dir = output_dir / person_id
        person_dir.mkdir(parents=True, exist_ok=True)

        written_paths = []
        for name, mask in self.masks.items():
            mask_path = person_dir / f"{name}.png"
            cv2.imwrite(str(mask_path), mask)
            written_paths.append(mask_path)

        summary_path = person_dir / "regions.json"
        with open(summary_path, "w") as f:
            json.dump({"person_id": person_id, "regions": list(self.masks.keys())}, f, indent=2)
        written_paths.append(summary_path)

        return written_paths
