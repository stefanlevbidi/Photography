"""
NoaKat Portrait Engine -- entry point.

    PHOTO -> ANALYSIS -> INTELLIGENT PROCESSING -> DENSITY PREPARATION -> EXPORT

NoaKat prepares a portrait for Astute Graphics Stipplism. It never produces
dots, points, or final artwork -- only a prepared density image and the
metadata Stipplism needs to convert it into a crystal canvas portrait.

Usage:
    python main.py <path-to-photo>

If no path is given, the first supported image found in input/ is used.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from config import settings
from modules.analyzer.image_analyzer import ImageAnalyzer
from modules.face.face_detector import FaceDetector
from modules.face.face_regions import FaceRegions
from modules.eyes.eye_engine import EyeEngine
from modules.skin.skin_engine import SkinEngine
from modules.hair.hair_engine import HairEngine
from modules.background.background_engine import BackgroundEngine
from modules.density.density_engine import DensityEngine
from modules.export.export_engine import ExportEngine


def find_input_image():
    for path in sorted(settings.INPUT_DIR.iterdir()):
        if path.suffix.lower() in settings.SUPPORTED_FORMATS:
            return path
    return None


def run(image_path):
    print(f"NoaKat Portrait Engine -- analyzing: {image_path}")

    # Stage 1: ANALYSIS
    analyzer = ImageAnalyzer(image_path)
    analysis = analyzer.process()
    analysis_path = analyzer.export(settings.OUTPUT_DIR / "analysis.json")
    print(f"  Analysis written to: {analysis_path}")

    if analysis["exposure_problems"]["problems"]:
        print(f"  Exposure problems detected: {analysis['exposure_problems']['problems']}")
    else:
        print("  No exposure problems detected.")

    # Stage 2: FACE DETECTION
    detector = FaceDetector(image_path)
    faces = detector.process()
    face_paths = detector.export(settings.OUTPUT_DIR)
    print(f"  Detected {len(faces)} face(s): {[f['person_id'] for f in faces]}")
    for path in face_paths:
        print(f"  Face data written to: {path}")

    # Stage 3: FACE REGIONS
    skin_masks = {}
    hair_masks = {}
    background_masks = {}
    for face in faces:
        regions = FaceRegions(image_path, face["box"])
        regions.process()
        region_paths = regions.export(person_id=face["person_id"])
        print(f"  Regions for {face['person_id']}: {list(regions.masks.keys())}")
        for path in region_paths:
            print(f"  Region data written to: {path}")
        skin_masks[face["person_id"]] = regions.masks["skin"]
        hair_masks[face["person_id"]] = regions.masks["hair"]
        background_masks[face["person_id"]] = regions.masks["background"]

    # Stage 4: EYE ENGINE
    for face in faces:
        eye_engine = EyeEngine(image_path, face["box"])
        eye_engine.process()
        eye_paths = eye_engine.export(person_id=face["person_id"])
        print(f"  Eyes for {face['person_id']}: {len(eye_engine.eyes)} detected")
        for path in eye_paths:
            print(f"  Eye data written to: {path}")

    # Stage 5: SKIN ENGINE
    skin_prepared = {}
    for face in faces:
        skin_engine = SkinEngine(image_path, skin_masks[face["person_id"]])
        skin_engine.analyze()
        skin_engine.process()
        skin_paths = skin_engine.export(person_id=face["person_id"])
        skin_prepared[face["person_id"]] = skin_engine.prepared_image
        print(f"  Skin prepared for {face['person_id']}: noise {skin_engine.noise_estimate:.2f}")
        for path in skin_paths:
            print(f"  Skin data written to: {path}")

    # Stage 6: HAIR ENGINE
    hair_prepared = {}
    for face in faces:
        hair_engine = HairEngine(image_path, hair_masks[face["person_id"]])
        hair_engine.analyze()
        hair_engine.process()
        hair_paths = hair_engine.export(person_id=face["person_id"])
        hair_prepared[face["person_id"]] = hair_engine.prepared_image
        print(
            f"  Hair prepared for {face['person_id']}: dark_mass="
            f"{hair_engine.dark_mass_ratio:.2f} highlight={hair_engine.highlight_ratio:.2f}"
        )
        for path in hair_paths:
            print(f"  Hair data written to: {path}")

    # Stage 7: BACKGROUND ENGINE
    background_prepared = {}
    for face in faces:
        background_engine = BackgroundEngine(image_path, background_masks[face["person_id"]])
        background_engine.analyze()
        background_engine.process()
        background_paths = background_engine.export(person_id=face["person_id"])
        background_prepared[face["person_id"]] = background_engine.prepared_image
        print(f"  Background flattened for {face['person_id']}")
        for path in background_paths:
            print(f"  Background data written to: {path}")

    # Stage 8: DENSITY ENGINE
    # Combine each engine's region-specific edit onto the original photo --
    # the masks are mutually exclusive, so this is a plain masked overlay.
    original_image = cv2.imread(str(image_path))
    for face in faces:
        person_id = face["person_id"]
        composite = original_image.copy()
        skin_mask = skin_masks[person_id]
        hair_mask = hair_masks[person_id]
        background_mask = background_masks[person_id]
        composite = np.where(cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR) > 0, skin_prepared[person_id], composite)
        composite = np.where(cv2.cvtColor(hair_mask, cv2.COLOR_GRAY2BGR) > 0, hair_prepared[person_id], composite)
        composite = np.where(
            cv2.cvtColor(background_mask, cv2.COLOR_GRAY2BGR) > 0, background_prepared[person_id], composite
        )
        composite = composite.astype(np.uint8)

        composite_path = settings.OUTPUT_DIR / "masks" / person_id / "composite.png"
        composite_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(composite_path), composite)

        density_engine = DensityEngine(composite_path, levels=settings.DENSITY_LEVELS_DEFAULT)
        density_engine.analyze()
        density_engine.process()
        density_paths = density_engine.export(person_id=person_id)
        print(f"  Density map prepared for {person_id}: {settings.DENSITY_LEVELS_DEFAULT} levels")
        for path in density_paths:
            print(f"  Density data written to: {path}")

    # Stage 9: EXPORT
    export_engine = ExportEngine(settings.OUTPUT_DIR)
    export_engine.analyze()
    export_engine.process()
    export_paths = export_engine.export()
    print(f"  NoaKat output ready for Stipplism: {export_engine.metadata}")
    for name, path in export_paths.items():
        print(f"  {name}: {path}")

    return analysis, faces


def main():
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        image_path = find_input_image()
        if image_path is None:
            print(f"No image provided and none found in {settings.INPUT_DIR}")
            sys.exit(1)

    run(image_path)


if __name__ == "__main__":
    main()
