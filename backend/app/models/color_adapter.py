"""Color adapter: a small residual head trained on top of frozen CLIP embeddings.

Responsibility: nudge a CLIP image embedding so that images sharing a
color sit closer together in embedding space, without touching CLIP's
own weights (those stay frozen -- see `ClipModel`). This is the
"local fine-tune" alternative to updating CLIP itself: cheap enough to
train on a CPU laptop because it only ever does matrix ops on cached
512-dim vectors, never a forward/backward pass through the CLIP
backbone.

Not wired into the search API yet -- this module + the training
script in `scripts/train_color_adapter.py` are the training-side
scaffold. Applying the trained adapter at index/query time is a
follow-up integration step.
"""

from __future__ import annotations

import torch
from torch import nn


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
