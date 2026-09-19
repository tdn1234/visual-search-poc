"""Embedding service.

Responsibility: turn raw image input (bytes from an HTTP upload, or a
path on disk) into a CLIP embedding vector. This is the only layer
that talks to `ClipModel` directly -- API routes and the indexing
script both go through here so image-loading/validation logic lives
in exactly one place.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.models.clip_model import ClipModel

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Generates CLIP embeddings from images."""

    def __init__(self, clip_model: ClipModel) -> None:
        """Store a reference to an already-loaded CLIP model.

        Args:
            clip_model: A `ClipModel` instance. Injected rather than
                constructed here so the (slow) model load happens
                exactly once, at application startup.
        """
        self._clip_model = clip_model

    def embed_image_bytes(self, image_bytes: bytes) -> list[float]:
        """Generate an embedding for raw image bytes (e.g. an upload).

        Args:
            image_bytes: Raw bytes of an image file (jpg/png/...).

        Returns:
            A 512-dimensional, L2-normalized embedding vector.

        Raises:
            ValueError: If the bytes do not represent a valid image.
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except UnidentifiedImageError as exc:
            raise ValueError(
                "Uploaded file is not a valid image (jpg/png/webp)."
            ) from exc

        return self._clip_model.encode_image(image)

    def embed_image_file(self, image_path: Path) -> list[float]:
        """Generate an embedding for an image stored on disk.

        Args:
            image_path: Path to a product image on the local filesystem.

        Returns:
            A 512-dimensional, L2-normalized embedding vector.

        Raises:
            FileNotFoundError: If `image_path` does not exist.
            ValueError: If the file exists but is not a valid image.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path)
            image.load()
        except UnidentifiedImageError as exc:
            raise ValueError(f"File is not a valid image: {image_path}") from exc

        return self._clip_model.encode_image(image)
