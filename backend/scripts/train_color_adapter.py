"""CLI script: train the ColorAdapter head on top of frozen CLIP embeddings.

This is the "local fine-tune" for color-awareness: CLIP itself is
never updated (that would need a GPU and thousands of examples). We
freeze CLIP, precompute embeddings for the synthetic training set from
`scripts/generate_training_data.py`, and train a small residual head
(`ColorAdapter`) with a CLIP-style contrastive loss so an image's
adapted embedding moves closer to its color/category caption's text
embedding and away from other captions in the batch.

Cheap enough for a CPU laptop: after the one-time embedding precompute
pass, every training step is just matrix ops on cached 512-dim
vectors.

Usage (run from `backend/`, with the venv active):

    python scripts/generate_training_data.py   # once, or to refresh the dataset
    python scripts/train_color_adapter.py
    python scripts/train_color_adapter.py --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import json
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

from app.models.clip_model import ClipModel  # noqa: E402
from app.models.color_adapter import ColorAdapter  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TRAINING_DIR = BACKEND_DIR / "data" / "training"
MANIFEST_FILE = TRAINING_DIR / "manifest.json"
EMBEDDING_CACHE_FILE = TRAINING_DIR / "embedding_cache.npz"
ADAPTER_OUTPUT_FILE = BACKEND_DIR / "data" / "color_adapter.pt"

VAL_FRACTION = 0.15
RANDOM_SEED = 42


def _load_manifest(manifest_file: Path) -> list[dict]:
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"No training manifest at {manifest_file}. "
            "Run 'python scripts/generate_training_data.py' first."
        )
    with manifest_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def _precompute_embeddings(
    manifest: list[dict], clip_model: ClipModel, cache_file: Path, use_cache: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (image_embeddings, text_embeddings, color_labels) for the whole manifest.

    Cached to disk (keyed only by manifest length, a cheap staleness
    check) since re-running CLIP over hundreds of images is the slow
    part of iterating on training hyperparameters.
    """
    if use_cache and cache_file.exists():
        cached = np.load(cache_file, allow_pickle=True)
        if int(cached["count"]) == len(manifest):
            logger.info("Loaded cached embeddings from %s", cache_file)
            return cached["image_embeddings"], cached["text_embeddings"], cached["colors"]
        logger.info("Cache at %s is stale (sample count changed); recomputing.", cache_file)

    logger.info("Encoding %d training images with frozen CLIP...", len(manifest))
    from PIL import Image

    image_embeddings = []
    for i, sample in enumerate(manifest):
        image_path = BACKEND_DIR / sample["image_path"]
        image = Image.open(image_path)
        image.load()
        image_embeddings.append(clip_model.encode_image(image))
        if (i + 1) % 100 == 0:
            logger.info("  encoded %d/%d images", i + 1, len(manifest))
    image_embeddings_arr = np.array(image_embeddings, dtype=np.float32)

    logger.info("Encoding %d captions with frozen CLIP text tower...", len(manifest))
    captions = [sample["caption"] for sample in manifest]
    text_embeddings_arr = np.array(clip_model.encode_text(captions), dtype=np.float32)

    colors_arr = np.array([sample["color"] for sample in manifest])

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_file,
        image_embeddings=image_embeddings_arr,
        text_embeddings=text_embeddings_arr,
        colors=colors_arr,
        count=len(manifest),
    )
    logger.info("Cached embeddings to %s", cache_file)
    return image_embeddings_arr, text_embeddings_arr, colors_arr


def _split_train_val(
    num_samples: int, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(num_samples)
    val_count = max(1, int(num_samples * val_fraction))
    return indices[val_count:], indices[:val_count]


def _contrastive_loss(
    image_embeddings: torch.Tensor, text_embeddings: torch.Tensor, temperature: float
) -> torch.Tensor:
    """Symmetric CLIP-style InfoNCE loss over one batch.

    Each image's matching caption is the positive; every other caption
    in the batch is a negative (and vice versa).
    """
    logits = image_embeddings @ text_embeddings.T / temperature
    targets = torch.arange(logits.shape[0], device=logits.device)
    loss_i2t = nn.functional.cross_entropy(logits, targets)
    loss_t2i = nn.functional.cross_entropy(logits.T, targets)
    return (loss_i2t + loss_t2i) / 2


@torch.no_grad()
def _evaluate_color_retrieval(
    adapter: ColorAdapter, image_embeddings: torch.Tensor, colors: np.ndarray, indices: np.ndarray
) -> float:
    """Fraction of val images whose nearest *other* val image shares its color.

    A quick, interpretable proxy for "did the adapter learn to cluster
    by color" -- not a substitute for evaluating against the real
    /search endpoint, just a fast in-loop sanity signal.
    """
    adapter.eval()
    subset = image_embeddings[indices]
    adapted = adapter(subset)
    similarity = adapted @ adapted.T
    similarity.fill_diagonal_(-float("inf"))
    nearest = similarity.argmax(dim=1).cpu().numpy()
    subset_colors = colors[indices]
    matches = subset_colors[nearest] == subset_colors
    adapter.train()
    return float(matches.mean())


def train(
    epochs: int = 30,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    temperature: float = 0.07,
    hidden_dim: int = 256,
    use_cache: bool = True,
) -> None:
    manifest = _load_manifest(MANIFEST_FILE)

    logger.info("Loading CLIP model (frozen; only used to precompute embeddings)...")
    clip_model = ClipModel()

    image_embeddings_np, text_embeddings_np, colors_np = _precompute_embeddings(
        manifest, clip_model, EMBEDDING_CACHE_FILE, use_cache
    )
    del clip_model  # frozen CLIP is no longer needed once embeddings are cached

    image_embeddings = torch.from_numpy(image_embeddings_np)
    text_embeddings = torch.from_numpy(text_embeddings_np)

    train_idx, val_idx = _split_train_val(len(manifest), VAL_FRACTION, RANDOM_SEED)
    logger.info("Train/val split: %d / %d samples", len(train_idx), len(val_idx))

    embedding_dim = image_embeddings.shape[1]
    adapter = ColorAdapter(embedding_dim=embedding_dim, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(adapter.parameters(), lr=learning_rate)

    baseline_accuracy = _evaluate_color_retrieval(adapter, image_embeddings, colors_np, val_idx)
    logger.info("Baseline (untrained adapter) val color-retrieval accuracy: %.3f", baseline_accuracy)

    start_time = time.perf_counter()
    rng = np.random.default_rng(RANDOM_SEED)

    for epoch in range(1, epochs + 1):
        epoch_indices = rng.permutation(train_idx)
        epoch_loss = 0.0
        num_batches = 0

        for start in range(0, len(epoch_indices), batch_size):
            batch_idx = epoch_indices[start : start + batch_size]
            if len(batch_idx) < 2:  # contrastive loss needs >=2 samples to have negatives
                continue

            batch_images = image_embeddings[batch_idx]
            batch_texts = text_embeddings[batch_idx]

            adapted_images = adapter(batch_images)
            loss = _contrastive_loss(adapted_images, batch_texts, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        if epoch % 5 == 0 or epoch == epochs:
            val_accuracy = _evaluate_color_retrieval(adapter, image_embeddings, colors_np, val_idx)
            logger.info(
                "epoch %3d/%d | loss %.4f | val color-retrieval accuracy %.3f",
                epoch,
                epochs,
                epoch_loss / max(num_batches, 1),
                val_accuracy,
            )

    elapsed = time.perf_counter() - start_time
    final_accuracy = _evaluate_color_retrieval(adapter, image_embeddings, colors_np, val_idx)
    logger.info(
        "Training done in %.1fs. val color-retrieval accuracy: baseline %.3f -> trained %.3f",
        elapsed,
        baseline_accuracy,
        final_accuracy,
    )

    ADAPTER_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": adapter.state_dict(),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
        },
        ADAPTER_OUTPUT_FILE,
    )
    logger.info("Saved adapter weights to %s", ADAPTER_OUTPUT_FILE)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ColorAdapter head on cached CLIP embeddings.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hidden-dim", type=int, default=256)
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
        temperature=args.temperature,
        hidden_dim=args.hidden_dim,
        use_cache=not args.no_cache,
    )


if __name__ == "__main__":
    main()
