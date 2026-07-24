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

from config import settings
from modules.analyzer.image_analyzer import ImageAnalyzer
from modules.face.face_detector import FaceDetector
from modules.face.face_regions import FaceRegions
from modules.eyes.eye_engine import EyeEngine
from modules.skin.skin_engine import SkinEngine


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
    for face in faces:
        regions = FaceRegions(image_path, face["box"])
        regions.process()
        region_paths = regions.export(person_id=face["person_id"])
        print(f"  Regions for {face['person_id']}: {list(regions.masks.keys())}")
        for path in region_paths:
            print(f"  Region data written to: {path}")
        skin_masks[face["person_id"]] = regions.masks["skin"]

    # Stage 4: EYE ENGINE
    for face in faces:
        eye_engine = EyeEngine(image_path, face["box"])
        eye_engine.process()
        eye_paths = eye_engine.export(person_id=face["person_id"])
        print(f"  Eyes for {face['person_id']}: {len(eye_engine.eyes)} detected")
        for path in eye_paths:
            print(f"  Eye data written to: {path}")

    # Stage 5: SKIN ENGINE
    for face in faces:
        skin_engine = SkinEngine(image_path, skin_masks[face["person_id"]])
        skin_engine.analyze()
        skin_engine.process()
        skin_paths = skin_engine.export(person_id=face["person_id"])
        print(f"  Skin prepared for {face['person_id']}: noise {skin_engine.noise_estimate:.2f}")
        for path in skin_paths:
            print(f"  Skin data written to: {path}")

    # Stages 6-8: hair/background, density, export.
    # Each has scaffolding under modules/ (analyze/process/export) but is not
    # yet implemented -- they land in later development passes.
    print(
        "  Remaining pipeline stages (hair, background, density, export) "
        "are scaffolded under modules/ and not yet implemented."
    )

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
