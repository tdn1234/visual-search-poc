"""Color adapter: a small residual head trained on top of frozen CLIP embeddings.

Responsibility: nudge a CLIP image embedding so that images sharing a
color sit closer together in embedding space, without touching CLIP's
own weights (those stay frozen -- see `ClipModel`). This is the
"local fine-tune" alternative to updating CLIP itself: cheap enough to
train on a CPU laptop because it only ever does matrix ops on cached
512-dim vectors, never a forward/backward pass through the CLIP
backbone.

Trained via `scripts/train_color_adapter.py`, which saves weights to
`backend/data/color_adapter.pt`. `load_color_adapter` below loads that
checkpoint for use by `EmbeddingService`, which applies it (when
present) to every embedding it produces -- both catalog images (at
index time) and the query image (at search time) -- so the two sides
stay in the same adapted space.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

logger = logging.getLogger(__name__)


class ColorAdapter(nn.Module):
    """Residual MLP head: embedding -> embedding, same dimensionality.

    Architecture is intentionally tiny (one hidden layer) since it's
    trained on a few hundred examples -- a bigger head would just
    overfit.
    """

    def __init__(self, embedding_dim: int = 512, hidden_dim: int = 256) -> None:
        """Build the adapter.

        Args:
            embedding_dim: Dimensionality of CLIP embeddings (512 for
                clip-vit-base-patch32).
            hidden_dim: Width of the bottleneck hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply the adapter with a residual connection, then re-normalize.

        Args:
            embeddings: `(batch, embedding_dim)` L2-normalized CLIP embeddings.

        Returns:
            `(batch, embedding_dim)` L2-normalized adapted embeddings.
        """
        adapted = embeddings + self.net(embeddings)
        return adapted / adapted.norm(p=2, dim=-1, keepdim=True)


def load_color_adapter(checkpoint_file: Path) -> ColorAdapter | None:
    """Load a trained `ColorAdapter` from disk, if one has been trained.

    Returns `None` (rather than raising) when no checkpoint exists yet,
    so the app can start up and serve plain-CLIP search before anyone
    has run `scripts/train_color_adapter.py` -- consistent with how
    `IndexingService.load_index` treats a missing embeddings.json.

    Args:
        checkpoint_file: Path to a `.pt` file saved by
            `scripts/train_color_adapter.py`.

    Returns:
        A `ColorAdapter` in eval mode, or `None` if `checkpoint_file`
        does not exist.
    """
    if not checkpoint_file.exists():
        logger.warning(
            "No color adapter checkpoint at %s. Search will use raw CLIP "
            "embeddings. Run 'python scripts/train_color_adapter.py' to train one.",
            checkpoint_file,
        )
        return None

    checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
    adapter = ColorAdapter(
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
    )
    adapter.load_state_dict(checkpoint["state_dict"])
    adapter.eval()
    logger.info("Loaded color adapter from %s", checkpoint_file)
    return adapter
