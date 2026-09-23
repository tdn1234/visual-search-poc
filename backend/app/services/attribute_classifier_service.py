"""Attribute classifier service.

Responsibility: figure out "what category/color is *in* this uploaded
photo". Powers the `match_category`/`match_color` checkboxes on
`POST /search` -- a separate concern from the color adapter (which
only re-ranks results) and from `catalog_filter` (which applies
whatever label this service predicts).

Category and color use different techniques, because plain zero-shot
CLIP classification turned out unreliable for category on this
project's synthetic placeholder catalog (verified: near-random
similarity scores across categories on a real test image) while it
worked fine for color (a literal pixel property, not shape/semantics):

- Color: zero-shot, via `ClipModel.classify` against text prompts.
  Generalizes automatically to whatever colors exist in the catalog,
  no training required.
- Category: a trained `CategoryClassifier`, if one is loaded (falls
  back to the same unreliable zero-shot approach, with a warning, if
  no checkpoint has been trained yet).
"""

from __future__ import annotations

import logging

from PIL import Image

from app.models.category_classifier import CategoryClassifier
from app.models.clip_model import ClipModel

logger = logging.getLogger(__name__)


class AttributeClassifierService:
    """Classifies an uploaded image's category and/or color."""

    def __init__(self, clip_model: ClipModel, category_classifier: CategoryClassifier | None = None) -> None:
        """Store references to an already-loaded CLIP model and, optionally, a trained category classifier.

        Args:
            clip_model: A `ClipModel` instance, shared with the rest
                of the app (injected, not constructed here).
            category_classifier: A trained `CategoryClassifier`, or
                `None` to fall back to (unreliable) zero-shot category
                classification.
        """
        self._clip_model = clip_model
        self._category_classifier = category_classifier

    def classify(
        self,
        image: Image.Image,
        category_labels: list[str] | None = None,
        color_labels: list[str] | None = None,
    ) -> tuple[str | None, str | None]:
        """Predict which of the given category/color labels best fit `image`.

        Encodes `image` at most once, regardless of whether one or
        both attributes are requested.

        Args:
            image: The uploaded image to classify.
            category_labels: Candidate categories to choose from (e.g.
                every distinct category in the catalog). `None` or
                empty skips category classification.
            color_labels: Candidate colors to choose from. `None` or
                empty skips color classification.

        Returns:
            `(predicted_category, predicted_color)`, each `None` if
            its label list wasn't given.
        """
        if not category_labels and not color_labels:
            return None, None

        image_embedding = self._clip_model.encode_image(image)

        predicted_category = (
            self._classify_category(image_embedding, category_labels) if category_labels else None
        )
        predicted_color = (
            self._clip_model.classify(image_embedding, self._color_prompts(color_labels))
            if color_labels
            else None
        )
        return predicted_category, predicted_color

    def _classify_category(self, image_embedding: list[float], category_labels: list[str]) -> str | None:
        if self._category_classifier is None:
            logger.warning(
                "match_category requested but no trained category classifier is loaded; "
                "falling back to zero-shot CLIP classification (unreliable on synthetic "
                "placeholder images -- run scripts/train_category_classifier.py)."
            )
            return self._clip_model.classify(image_embedding, self._category_prompts(category_labels))

        predicted = self._category_classifier.predict(image_embedding)
        # Guard against the trained classifier predicting a label that
        # isn't actually one of the categories present right now (e.g.
        # the catalog changed since it was trained).
        return predicted if predicted in category_labels else None

    @staticmethod
    def _category_prompts(categories: list[str]) -> dict[str, str]:
        return {category: f"a photo of {category.lower()}" for category in categories}

    @staticmethod
    def _color_prompts(colors: list[str]) -> dict[str, str]:
        return {color: f"a photo of something {color.lower()} colored" for color in colors}
