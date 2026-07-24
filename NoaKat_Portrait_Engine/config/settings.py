"""
Central configuration for the NoaKat Portrait Engine.

NoaKat prepares portraits for Astute Graphics Stipplism. It never generates
dots, points, or final artwork itself -- every value in this file tunes the
*preparation* pipeline (analysis, masking, density mapping, export).
"""

from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MASKS_DIR = OUTPUT_DIR / "masks"

SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# ------------------------------------------------------------------
# Density engine (the heart of NoaKat)
# ------------------------------------------------------------------
# A photo carries 256 grayscale levels. Stippling only needs a controlled
# band of meaningful density steps -- not full tonal range, not a flat poster.
DENSITY_LEVELS_MIN = 25
DENSITY_LEVELS_MAX = 40
DENSITY_LEVELS_DEFAULT = 32

# ------------------------------------------------------------------
# Exposure / analysis thresholds
# ------------------------------------------------------------------
BRIGHTNESS_LOW_THRESHOLD = 70       # mean luminance below this = underexposed
BRIGHTNESS_HIGH_THRESHOLD = 190     # mean luminance above this = overexposed
CONTRAST_LOW_THRESHOLD = 35         # std-dev below this = flat / low contrast
CLIPPED_SHADOW_RATIO = 0.02         # fraction of pixels near 0 considered clipped
CLIPPED_HIGHLIGHT_RATIO = 0.02      # fraction of pixels near 255 considered clipped
MIN_RECOMMENDED_RESOLUTION = (1500, 1500)  # px, for crystal canvas quality

# ------------------------------------------------------------------
# Background engine
# ------------------------------------------------------------------
BACKGROUND_TARGET_RGB = (0, 0, 0)   # pure black, no gray contamination

# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------
FINAL_DENSITY_MAP_NAME = "final_density_map.png"
INVERTED_DENSITY_MAP_NAME = "inverted_density_map.png"
METADATA_NAME = "metadata.json"
