"""Similarity service.

Responsibility: given a query embedding and a catalog of stored
product embeddings, rank the catalog by visual similarity and return
the top-K matches.

Uses scikit-learn's `cosine_similarity` for clarity -- with a catalog
of 20-50 products this is a single, near-instant matrix operation, so
no vector database is needed for the POC.
"""

from __future__ import annotations

import logging

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import TOP_K_RESULTS
from app.schemas.search import ProductRecord, SearchResult

logger = logging.getLogger(__name__)


class SimilarityService:
    """Ranks catalog products against a query embedding."""

    def rank_products(
        self,
        query_embedding: list[float],
        catalog: list[ProductRecord],
        top_k: int = TOP_K_RESULTS,
    ) -> list[SearchResult]:
        """Return the `top_k` catalog products most similar to the query.

        How ranking works:
            1. Every stored embedding is already L2-normalized (done in
               `ClipModel.encode_image`), and so is the query embedding.
            2. `cosine_similarity` computes, for each catalog embedding,
               the cosine of the angle between it and the query vector.
               The result is a score in [-1, 1], where 1 means the
               images are visually identical according to CLIP, and
               values near 0 mean unrelated.
            3. We sort all catalog products by that score, descending,
               and keep the first `top_k`.

        Args:
            query_embedding: The embedding of the uploaded search image.
            catalog: All products currently indexed, each with its
                precomputed embedding.
            top_k: How many results to return.

        Returns:
            A list of `SearchResult`, sorted by descending similarity
            score, of length `min(top_k, len(catalog))`.

        Raises:
            ValueError: If the catalog is empty.
        """
        if not catalog:
            raise ValueError(
                "Product catalog is empty. Run 'python scripts/build_embeddings.py' "
                "to index products before searching."
            )

        query_vector = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        catalog_matrix = np.array(
            [product.embedding for product in catalog], dtype=np.float32
        )

        # Shape: (1, num_products) -> flatten to (num_products,)
        similarity_scores = cosine_similarity(query_vector, catalog_matrix).flatten()

        ranked_indices = np.argsort(-similarity_scores)[:top_k]

        results = [
            SearchResult(
                sku=catalog[idx].sku,
                name=catalog[idx].name,
                price=catalog[idx].price,
                category=catalog[idx].category,
                score=round(float(similarity_scores[idx]), 4),
            )
            for idx in ranked_indices
        ]
        return results
