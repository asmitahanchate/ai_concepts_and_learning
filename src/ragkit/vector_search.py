"""
vector_search.py: the simplest possible vector database: a NumPy matrix.

"Brute force" (exact) search: compare the query against EVERY stored vector.
Because all vectors have length 1, cosine similarity = dot product, so one
matrix multiplication scores the whole corpus at once.

This is perfect for a few thousand chunks. At millions of chunks it gets too slow,
which is why real vector databases use approximate indexes like HNSW (Project 03).
"""

from __future__ import annotations

import numpy as np


class VectorIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors  # shape (n_chunks, dim), each row length 1

    def search(self, query_vector: np.ndarray, k: int = 5) -> list[tuple[int, float]]:
        """Return the top-k (row_index, similarity) pairs, most similar first."""
        sims = self.vectors @ query_vector  # one similarity score per chunk
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]  # the k best, unordered (fast: no full sort)
        top = top[np.argsort(-sims[top])]  # now sort just those k
        return [(int(i), float(sims[i])) for i in top]
