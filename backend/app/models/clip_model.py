"""CLIP model wrapper.

Responsibility: load the pretrained CLIP model/processor exactly once
and expose image encoding, text encoding, and zero-shot classification
built on top of them -- all as plain Python types (lists of floats,
strings), never tensors.

Nothing else in the codebase should import `transformers` or `torch`
directly for inference -- everything goes through this class so the
ML framework can be swapped later (e.g. for a different model or an
ONNX runtime) without touching services or the API layer.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import CLIP_MODEL_NAME, DEVICE

logger = logging.getLogger(__name__)


class ClipModel:
    """Loads openai/clip-vit-base-patch32 and encodes images to vectors.

    This class is intentionally a thin wrapper: it owns the heavyweight
    model/processor objects and exposes one clear method,
    `encode_image`, so callers never need to know about tensors,
    devices, or normalization details.
    """

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: str = DEVICE) -> None:
        """Load the CLIP model and processor into memory.

        Args:
            model_name: Hugging Face model id to load.
            device: torch device string, e.g. "cpu" or "cuda".

        Raises:
            RuntimeError: If the model cannot be downloaded/loaded
                (e.g. no internet on first run, or a corrupted cache).
        """
        self.device = device
        try:
            logger.info("Loading CLIP model '%s' on device '%s'...", model_name, device)
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("CLIP model loaded successfully.")
        except Exception as exc:  # noqa: BLE001 - we want to wrap any load failure
            raise RuntimeError(
                f"Failed to load CLIP model '{model_name}'. "
                "Check your internet connection (first run downloads the "
                "model from Hugging Face) and that 'torch' and "
                "'transformers' are installed correctly."
            ) from exc

    @torch.no_grad()
    def encode_image(self, image: Image.Image) -> list[float]:
        """Encode a PIL image into a normalized CLIP embedding.

        The embedding is L2-normalized so that a plain dot product
        between two embeddings is equivalent to cosine similarity.

        Args:
            image: A PIL Image, already opened and converted (RGB).

        Returns:
            A list of 512 floats (the embedding dimension for
            clip-vit-base-patch32).

        Raises:
            ValueError: If the image cannot be processed by CLIP.
        """
        try:
            rgb_image = image.convert("RGB")
            inputs = self.processor(images=rgb_image, return_tensors="pt")
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            image_features = self.model.get_image_features(**inputs)

            # L2-normalize so cosine similarity == dot product later on.
            norm = image_features.norm(p=2, dim=-1, keepdim=True)
            normalized_features = image_features / norm

            embedding: np.ndarray = normalized_features.squeeze(0).cpu().numpy()
            return embedding.tolist()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"Could not encode image with CLIP: {exc}") from exc

    @torch.no_grad()
    def encode_text(self, texts: list[str]) -> list[list[float]]:
        """Encode a batch of text strings into normalized CLIP embeddings.

        Used for adapter training (contrasting an image embedding
        against a caption like "a brown bag") and for zero-shot
        classification (see `classify`).

        Args:
            texts: A batch of raw strings.

        Returns:
            A list of L2-normalized 512-dim embeddings, one per input string.
        """
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        text_features = self.model.get_text_features(**inputs)
        norm = text_features.norm(p=2, dim=-1, keepdim=True)
        normalized_features = text_features / norm

        embedding: np.ndarray = normalized_features.cpu().numpy()
        return embedding.tolist()

    @torch.no_grad()
    def classify(self, image_embedding: list[float], candidate_texts: dict[str, str]) -> str:
        """Zero-shot classify a precomputed image embedding against text prompts.

        Standard CLIP zero-shot classification: encode every candidate
        prompt, and return whichever one is most similar (cosine, via
        dot product since both sides are L2-normalized) to
        `image_embedding`. This always uses the *raw* CLIP embedding
        space, never the color adapter's -- the adapter is trained to
        help ranking, not classification, and mixing the two spaces
        here would give meaningless similarities.

        Args:
            image_embedding: A normalized embedding from `encode_image`
                (not one that's been through `ColorAdapter`).
            candidate_texts: Mapping of label -> prompt text, e.g.
                `{"Shoes": "a photo of shoes"}`.

        Returns:
            Whichever key of `candidate_texts` has the highest-scoring
            prompt.
        """
        labels = list(candidate_texts.keys())
        prompts = list(candidate_texts.values())

        image_vector = torch.tensor(image_embedding, dtype=torch.float32)
        text_vectors = torch.tensor(self.encode_text(prompts), dtype=torch.float32)

        similarities = text_vectors @ image_vector
        best_index = int(similarities.argmax())
        return labels[best_index]
