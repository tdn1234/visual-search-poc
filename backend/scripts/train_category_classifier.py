"""CLI script: train the CategoryClassifier head on top of frozen CLIP embeddings.

Mirrors `train_color_adapter.py`'s approach (freeze CLIP, train a
small head on cached embeddings) but for category classification.
Needed because plain zero-shot CLIP classification turned out
unreliable on this catalog's synthetic placeholder images -- verified
similarity scores across categories were within 0.02 of each other
for a real test query (near-random), while color survived zero-shot
fine. A trained classifier on the same synthetic dataset fixes that.

Reuses `train_color_adapter`'s manifest loading and embedding
precompute/cache helpers directly (same training set, same cache
file) rather than duplicating them.

Usage (run from `backend/`, with the venv active):

    python scripts/generate_training_data.py   # once, or to refresh the dataset
    python scripts/train_category_classifier.py
    python scripts/train_category_classifier.py --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.models.category_classifier import CategoryClassifier  # noqa: E402
from app.models.clip_model import ClipModel  # noqa: E402
from scripts.train_color_adapter import (  # noqa: E402
    EMBEDDING_CACHE_FILE,
    MANIFEST_FILE,
    RANDOM_SEED,
    VAL_FRACTION,
    _load_manifest,
    _precompute_embeddings,
    _split_train_val,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

CLASSIFIER_OUTPUT_FILE = BACKEND_DIR / "data" / "category_classifier.pt"

# Maps the synthetic training set's per-shape category word
# (generate_training_data.py's CATEGORIES) to this project's actual
# catalog-level category names (see catalog/*/metadata.json).
TRAINING_TO_CATALOG_CATEGORY = {
    "shoe": "Shoes",
    "bag": "Bags",
    "hat": "Accessories",
    "shirt": "Apparel",
    "watch": "Accessories",
}


@torch.no_grad()
def _accuracy(classifier: CategoryClassifier, embeddings: torch.Tensor, targets: torch.Tensor, indices: np.ndarray) -> float:
    classifier.eval()
    logits = classifier(embeddings[indices])
    predictions = logits.argmax(dim=-1)
    classifier.train()
    return float((predictions == targets[indices]).float().mean())


def train(
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    use_cache: bool = True,
) -> None:
    manifest = _load_manifest(MANIFEST_FILE)

    logger.info("Loading CLIP model (frozen; only used to precompute embeddings)...")
    clip_model = ClipModel()

    image_embeddings_np, _text_embeddings_np, _colors_np = _precompute_embeddings(
        manifest, clip_model, EMBEDDING_CACHE_FILE, use_cache
    )
    del clip_model  # frozen CLIP is no longer needed once embeddings are cached

    category_labels = sorted(set(TRAINING_TO_CATALOG_CATEGORY.values()))
    label_to_index = {label: index for index, label in enumerate(category_labels)}
    targets_np = np.array(
        [label_to_index[TRAINING_TO_CATALOG_CATEGORY[sample["category"]]] for sample in manifest],
        dtype=np.int64,
    )

    image_embeddings = torch.from_numpy(image_embeddings_np)
    targets = torch.from_numpy(targets_np)

    train_idx, val_idx = _split_train_val(len(manifest), VAL_FRACTION, RANDOM_SEED)
    logger.info("Train/val split: %d / %d samples", len(train_idx), len(val_idx))
    logger.info("Category labels: %s", category_labels)

    embedding_dim = image_embeddings.shape[1]
    classifier = CategoryClassifier(embedding_dim=embedding_dim, category_labels=category_labels)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    baseline_accuracy = _accuracy(classifier, image_embeddings, targets, val_idx)
    logger.info("Baseline (untrained classifier) val accuracy: %.3f", baseline_accuracy)

    start_time = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)

    for epoch in range(1, epochs + 1):
        epoch_indices = rng.permutation(train_idx)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, len(epoch_indices), batch_size):
            batch_idx = epoch_indices[start : start + batch_size]
            batch_images = image_embeddings[batch_idx]
            batch_targets = targets[batch_idx]

            logits = classifier(batch_images)
            loss = nn.functional.cross_entropy(logits, batch_targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            val_accuracy = _accuracy(classifier, image_embeddings, targets, val_idx)
            logger.info(
                "epoch %3d/%d | loss %.4f | val accuracy %.3f",
                epoch,
                epochs,
                epoch_loss / max(num_batches, 1),
                val_accuracy,
            )

    elapsed = time.perf_counter() - start_time
    final_accuracy = _accuracy(classifier, image_embeddings, targets, val_idx)
    logger.info(
        "Training done in %.1fs. val accuracy: baseline %.3f -> trained %.3f",
        elapsed,
        baseline_accuracy,
        final_accuracy,
    )

    CLASSIFIER_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": classifier.state_dict(),
            "embedding_dim": embedding_dim,
            "category_labels": category_labels,
        },
        CLASSIFIER_OUTPUT_FILE,
    )
    logger.info("Saved category classifier to %s", CLASSIFIER_OUTPUT_FILE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CategoryClassifier head on cached CLIP embeddings.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Recompute CLIP embeddings instead of reusing embedding_cache.npz.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
