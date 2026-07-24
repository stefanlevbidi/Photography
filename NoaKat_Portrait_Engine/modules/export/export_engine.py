"""
Export Engine -- packages NoaKat's output for Astute Graphics Stipplism.

Writes final_density_map.png, inverted_density_map.png, and metadata.json
at the top of output/, alongside the per-person masks/analysis every
earlier engine already wrote under output/masks/<person_id>/. NoaKat's job
ends here -- Stipplism takes this output and creates the dots.

For a single detected person (the common portrait case), the top-level
density map is simply that person's density map, promoted. For a group
portrait with several people, the first detected person (left-to-right,
per FaceDetector's ordering) is promoted as the primary output, since
merging several people's independently-segmented density maps into one
consistent scene-level map is a further step this module doesn't attempt --
every person's own density map remains available under their masks/
subfolder regardless.
"""

import json
import shutil
from pathlib import Path

from config import settings


class ExportEngine:
    """Assembles and writes the final NoaKat output package."""

    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else settings.OUTPUT_DIR
        self.faces_data = None
        self.metadata = None

    # ------------------------------------------------------------------
    def analyze(self):
        """Verify that all required inputs (density maps, masks) are present."""
        masks_dir = self.output_dir / "masks"
        if not masks_dir.exists():
            raise FileNotFoundError(f"No masks directory found at {masks_dir}; run the pipeline first")

        person_dirs = sorted(p for p in masks_dir.iterdir() if p.is_dir() and p.name.startswith("person_"))
        if not person_dirs:
            raise FileNotFoundError(f"No person data found under {masks_dir}")

        found = []
        for person_dir in person_dirs:
            density_dir = person_dir / "density"
            density_path = density_dir / settings.FINAL_DENSITY_MAP_NAME
            inverted_path = density_dir / settings.INVERTED_DENSITY_MAP_NAME
            if not (density_path.exists() and inverted_path.exists()):
                continue

            levels = None
            curve_path = density_dir / "tonal_curve.json"
            if curve_path.exists():
                with open(curve_path) as f:
                    levels = json.load(f).get("levels")

            found.append({
                "person_id": person_dir.name,
                "density_path": density_path,
                "inverted_path": inverted_path,
                "levels": levels,
            })

        if not found:
            raise FileNotFoundError("No completed density maps found; run the Density Engine stage first")

        self.faces_data = found
        return self.faces_data

    # ------------------------------------------------------------------
    def process(self):
        """Assemble the output package (density maps, masks, metadata)."""
        if self.faces_data is None:
            self.analyze()

        primary = self.faces_data[0]
        final_density_path = self.output_dir / settings.FINAL_DENSITY_MAP_NAME
        final_inverted_path = self.output_dir / settings.INVERTED_DENSITY_MAP_NAME
        self.output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(primary["density_path"], final_density_path)
        shutil.copyfile(primary["inverted_path"], final_inverted_path)

        self.metadata = {
            "faces": len(self.faces_data),
            "density_levels": primary["levels"],
            "regions": True,
            "stipplism_ready": True,
        }
        return self.metadata

    # ------------------------------------------------------------------
    def export(self):
        """Write everything to output/ in the NoaKat directory layout."""
        if self.metadata is None:
            self.process()

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)

        return {
            "final_density_map": self.output_dir / settings.FINAL_DENSITY_MAP_NAME,
            "inverted_density_map": self.output_dir / settings.INVERTED_DENSITY_MAP_NAME,
            "masks_dir": self.output_dir / "masks",
            "metadata": metadata_path,
        }
