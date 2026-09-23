"""CLI script: merge real product photos into the training manifest.

The synthetic dataset (`generate_training_data.py`) is enough to teach
the category classifier the *shapes* used by this project's own
placeholder catalog, but it doesn't generalize to real photos -- a
real image lands somewhere in CLIP's embedding space the classifier
never saw during training (verified: it misclassified a real shoe
photo as "Accessories", while raw zero-shot CLIP got it right). Real
training examples are the actual fix.

Usage: drop photos into `backend/data/real_training/<Category>/`,
where `<Category>` is one of this project's real catalog categories
(Shoes, Bags, Accessories, Apparel -- case-sensitive, must match
exactly). Optionally prefix a filename with a color name and a hyphen
(e.g. `red-my-sneaker.jpg`) to also label its color for color-adapter
training; photos without a recognized color prefix still count for
category-classifier training, just not color-adapter training.

    python scripts/generate_training_data.py   # first, if not already run
    python scripts/import_real_photos.py
    python scripts/train_category_classifier.py --no-cache
    python scripts/train_color_adapter.py --no-cache   # optional
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

REAL_PHOTOS_DIR = BACKEND_DIR / "data" / "real_training"
TRAINING_DIR = BACKEND_DIR / "data" / "training"
MANIFEST_FILE = TRAINING_DIR / "manifest.json"
IMAGES_DIR = TRAINING_DIR / "images"

# Must match the real catalog's actual category names exactly --
# these are used as-is (not run through TRAINING_TO_CATALOG_CATEGORY,
# which only applies to the synthetic generator's per-shape words).
VALID_CATEGORIES = {"Shoes", "Bags", "Accessories", "Apparel"}

# Recognized color prefixes, e.g. "red-my-sneaker.jpg" -> color "red".
# Keep in sync with generate_training_data.py's COLORS so imported
# photos and synthetic ones share a color vocabulary.
KNOWN_COLORS = {
    "red", "blue", "green", "black", "white", "brown",
    "silver", "yellow", "orange", "purple", "pink", "navy",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _parse_color_prefix(filename: str) -> str | None:
    """Return the leading color name from a filename like 'red-shoe.jpg', or None."""
    stem = Path(filename).stem
    match = re.match(r"^([a-zA-Z]+)-", stem)
    if match and match.group(1).lower() in KNOWN_COLORS:
        return match.group(1).lower()
    return None


def import_real_photos() -> list[dict]:
    """Copy every photo under `real_training/<Category>/` into the training set.

    Idempotent: re-running only imports photos not already present in
    the manifest (matched by destination path), so it's safe to run
    again after adding more photos.

    Returns:
        The list of newly-added manifest entries (empty if nothing new).

    Raises:
        FileNotFoundError: If `real_training/` or the manifest doesn't
            exist yet.
    """
    if not REAL_PHOTOS_DIR.exists():
        raise FileNotFoundError(
            f"No real photos directory at {REAL_PHOTOS_DIR}. Create "
            f"backend/data/real_training/<Category>/ (Shoes, Bags, Accessories, "
            f"or Apparel) and drop photos into it."
        )

    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"No training manifest at {MANIFEST_FILE}. "
            "Run 'python scripts/generate_training_data.py' first."
        )

    with MANIFEST_FILE.open("r", encoding="utf-8") as file:
        manifest: list[dict] = json.load(file)
    existing_paths = {sample["image_path"] for sample in manifest}

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    new_entries: list[dict] = []

    for category_dir in sorted(REAL_PHOTOS_DIR.iterdir()):
        if not category_dir.is_dir():
            continue
        category = category_dir.name
        if category not in VALID_CATEGORIES:
            logger.warning(
                "Skipping '%s': not one of %s (rename the folder to match exactly)",
                category, sorted(VALID_CATEGORIES),
            )
            continue

        for photo_path in sorted(category_dir.iterdir()):
            if photo_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue

            dest_name = f"real-{category.lower()}-{photo_path.stem}{photo_path.suffix.lower()}"
            dest_path = IMAGES_DIR / dest_name
            dest_rel = str(dest_path.relative_to(BACKEND_DIR))

            if dest_rel in existing_paths:
                continue  # already imported

            shutil.copyfile(photo_path, dest_path)

            color = _parse_color_prefix(photo_path.name)
            caption = f"a {color} {category.lower()}" if color else f"a photo of {category.lower()}"

            entry = {"image_path": dest_rel, "category": category, "color": color, "caption": caption}
            manifest.append(entry)
            new_entries.append(entry)
            existing_paths.add(dest_rel)
            logger.info("Imported %s (category=%s, color=%s)", photo_path.name, category, color)

    with MANIFEST_FILE.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    logger.info(
        "Imported %d new real photo(s). Manifest now has %d samples total (%s).",
        len(new_entries), len(manifest), MANIFEST_FILE,
    )
    return new_entries


def main() -> None:
    import_real_photos()


if __name__ == "__main__":
    main()
