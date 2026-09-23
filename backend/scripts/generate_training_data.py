"""CLI script: generate a small synthetic image/color dataset for adapter training.

Why synthetic: the real product catalog only has 8 photos (not nearly
enough to train anything), and this POC runs offline on a laptop with
no budget for downloading/labeling a large external image set. So we
generate our own small dataset of colored shapes, in the same visual
style as the existing catalog placeholders (a solid-colored silhouette
on an off-white background), varying color, category, size, and
position so the adapter sees more than one example per class.

This is meant to be re-run any time you want a fresh/larger training
set. It never touches the real `catalog/` directory.

Usage (run from `backend/`, with the venv active):

    python scripts/generate_training_data.py
"""

from __future__ import annotations

import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# --- Output layout ----------------------------------------------------------
TRAINING_DIR = BACKEND_DIR / "data" / "training"
IMAGES_DIR = TRAINING_DIR / "images"
MANIFEST_FILE = TRAINING_DIR / "manifest.json"

# --- Dataset definition ------------------------------------------------------
IMAGE_SIZE = 400
BACKGROUND = (250, 240, 230)
SAMPLES_PER_COMBO = 10
RANDOM_SEED = 42

# Color name -> base RGB. Kept close to common product-color vocabulary
# so captions read naturally to CLIP's text tower (e.g. "a brown bag").
COLORS: dict[str, tuple[int, int, int]] = {
    "red": (200, 30, 30),
    "blue": (30, 80, 200),
    "green": (40, 140, 70),
    "black": (25, 25, 25),
    "white": (245, 245, 245),
    "brown": (110, 70, 40),
    "silver": (180, 180, 185),
    "yellow": (230, 195, 40),
    "orange": (220, 120, 30),
    "purple": (120, 50, 150),
    "pink": (230, 130, 170),
    "navy": (25, 35, 90),
}

# Category word -> shape-drawing function. Each draws a simple silhouette
# roughly evoking the product type, filled with the given jittered color.
def _draw_shoe(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    draw.ellipse(box, fill=fill)


def _draw_bag(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=30, fill=fill)


def _draw_hat(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    draw.pieslice(box, start=180, end=360, fill=fill)


def _draw_shirt(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    width = x1 - x0
    draw.rectangle((x0 + width * 0.2, y0, x1 - width * 0.2, y1), fill=fill)
    draw.polygon([(x0, y0 + 20), (x0 + width * 0.2, y0), (x0 + width * 0.2, y0 + 60)], fill=fill)
    draw.polygon([(x1, y0 + 20), (x1 - width * 0.2, y0), (x1 - width * 0.2, y0 + 60)], fill=fill)


def _draw_watch(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    draw.rectangle((cx - 15, y0 - 30, cx + 15, y1 + 30), fill=fill)
    draw.ellipse(box, fill=fill)


DrawFn = Callable[[ImageDraw.ImageDraw, tuple[float, float, float, float], tuple[int, int, int]], None]

CATEGORIES: dict[str, DrawFn] = {
    "shoe": _draw_shoe,
    "bag": _draw_bag,
    "hat": _draw_hat,
    "shirt": _draw_shirt,
    "watch": _draw_watch,
}


def _jitter_color(rgb: tuple[int, int, int], amount: int = 12) -> tuple[int, int, int]:
    """Nudge each channel by a small random amount, clamped to [0, 255]."""
    return tuple(max(0, min(255, channel + random.randint(-amount, amount))) for channel in rgb)


def _jitter_box(size: int = IMAGE_SIZE, base_extent: int = 220) -> tuple[int, int, int, int]:
    """Return a randomly sized/positioned bounding box roughly centered in the canvas."""
    extent = base_extent + random.randint(-30, 30)
    cx = size / 2 + random.randint(-25, 25)
    cy = size / 2 + random.randint(-25, 25)
    half = extent / 2
    return (cx - half, cy - half * 0.7, cx + half, cy + half * 0.7)


def generate_dataset(
    output_dir: Path = IMAGES_DIR,
    manifest_file: Path = MANIFEST_FILE,
    samples_per_combo: int = SAMPLES_PER_COMBO,
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Generate the synthetic image/color/caption dataset and write its manifest.

    Args:
        output_dir: Directory to write generated .jpg images into.
        manifest_file: Where to write the manifest.json describing each sample.
        samples_per_combo: How many jittered variants per (category, color) pair.
        seed: Random seed, for reproducible datasets.

    Returns:
        The manifest as a list of dicts: {image_path, category, color, caption}.
    """
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    sample_index = 0

    for category, draw_fn in CATEGORIES.items():
        for color_name, base_rgb in COLORS.items():
            for _ in range(samples_per_combo):
                image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), BACKGROUND)
                draw = ImageDraw.Draw(image)
                box = _jitter_box()
                fill = _jitter_color(base_rgb)
                draw_fn(draw, box, fill)

                filename = f"{category}-{color_name}-{sample_index:04d}.jpg"
                image_path = output_dir / filename
                image.save(image_path, quality=90)

                manifest.append(
                    {
                        "image_path": str(image_path.relative_to(BACKEND_DIR)),
                        "category": category,
                        "color": color_name,
                        "caption": f"a {color_name} {category}",
                    }
                )
                sample_index += 1

    with manifest_file.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    logger.info(
        "Generated %d synthetic training images (%d categories x %d colors x %d samples) -> %s",
        len(manifest),
        len(CATEGORIES),
        len(COLORS),
        samples_per_combo,
        output_dir,
    )
    logger.info("Manifest written to %s", manifest_file)
    return manifest


def main() -> None:
    generate_dataset()


if __name__ == "__main__":
    main()
