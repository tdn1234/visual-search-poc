"""Category classifier: a small linear head trained on frozen CLIP embeddings.

Responsibility: predict which catalog category an uploaded image
belongs to. Exists because plain zero-shot CLIP classification (text
prompts like "a photo of shoes" compared to the image) turned out to
be unreliable on this project's catalog -- verified: on a test query,
similarity scores across all four categories were bunched within 0.02
of each other, essentially noise, because the catalog's images are
abstract solid-color placeholder shapes, not real photos CLIP was
pretrained on. Color survives zero-shot fine (it's a literal pixel
property); category needed an actual trained classifier, same as the
color adapter needed training rather than relying on raw CLIP.

Trained via `scripts/train_category_classifier.py`, which saves
weights to `backend/data/category_classifier.pt`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class CategoryClassifier(nn.Module):
    """Linear classifier: raw CLIP embedding -> one of a fixed set of category labels."""

    def __init__(self, embedding_dim: int, category_labels: list[str]) -> None:
        """Build the classifier.

        Args:
            embedding_dim: Dimensionality of CLIP embeddings (512 for
                clip-vit-base-patch32).
            category_labels: The category names this classifier
                predicts among, in the order matching its output logits.
        """
        super().__init__()
        self.category_labels = category_labels
        self.linear = nn.Linear(embedding_dim, len(category_labels))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return raw logits, shape `(batch, len(category_labels))`."""
        return self.linear(embeddings)

    def predict(self, image_embedding: list[float]) -> str:
        """Predict the single best-matching category for one embedding.

        Args:
            image_embedding: A raw (non-color-adapted) CLIP image
                embedding from `ClipModel.encode_image`.

        Returns:
            The highest-scoring entry of `self.category_labels`.
        """
        with torch.no_grad():
            vector = torch.tensor(image_embedding, dtype=torch.float32).unsqueeze(0)
            logits = self.forward(vector)
            index = int(logits.argmax(dim=-1).item())
        return self.category_labels[index]


def load_category_classifier(checkpoint_file: Path) -> CategoryClassifier | None:
    """Load a trained `CategoryClassifier` from disk, if one has been trained.

    Returns `None` (rather than raising) when no checkpoint exists yet,
    consistent with `load_color_adapter` -- callers should fall back
    to a degraded (or skipped) behavior rather than fail startup.

    Args:
        checkpoint_file: Path to a `.pt` file saved by
            `scripts/train_category_classifier.py`.

    Returns:
        A `CategoryClassifier` in eval mode, or `None` if
        `checkpoint_file` does not exist.
    """
    if not checkpoint_file.exists():
        logger.warning(
            "No category classifier checkpoint at %s. match_category will fall back "
            "to zero-shot CLIP classification (unreliable on synthetic placeholder "
            "images). Run 'python scripts/train_category_classifier.py' to train one.",
            checkpoint_file,
        )
        return None

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    classifier = CategoryClassifier(
        embedding_dim=checkpoint["embedding_dim"],
        category_labels=checkpoint["category_labels"],
    )
    classifier.load_state_dict(checkpoint["state_dict"])
    classifier.eval()
    logger.info("Loaded category classifier from %s", checkpoint_file)
    return classifier
