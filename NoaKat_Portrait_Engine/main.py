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

    # Stages 2-5: face/regions, eyes/skin/hair/background, density, export.
    # Each has scaffolding under modules/ (analyze/process/export) but is not
    # yet implemented -- they land in later development passes.
    print(
        "  Remaining pipeline stages (face detection, regions, eyes, skin, "
        "hair, background, density, export) are scaffolded under modules/ "
        "and not yet implemented."
    )

    return analysis


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
