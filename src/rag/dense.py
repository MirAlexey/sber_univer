"""Плотный (семантический) поиск на эмбеддингах.

Индекс строится поверх эмбеддера с методом embed(text): в бою это
GigaChatEmbedder (обёртка над GigaChatEmbeddings), в офлайн-тестах — HashEmbedder.
Индекс — матрица нормализованных векторов NumPy с косинусной близостью, без
внешней векторной БД.
"""

from __future__ import annotations

import numpy as np



class DenseIndex:
    """Матрица эмбеддингов с косинусным поиском по запросу."""

    def __init__(self, embedder):
        self.embedder = embedder
        self._matrix: np.ndarray | None = None
        self._dim = 0

    def fit(self, texts: list[str]) -> None:
        vectors = [np.asarray(self.embedder.embed(t), dtype=np.float64) for t in texts]
        matrix = np.stack(vectors) if vectors else np.zeros((0, 764), dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._matrix = matrix
        self._dim = int(matrix.shape[1])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        if self._matrix is None or self._matrix.shape[0] == 0:
            return []
        q = np.asarray(self.embedder.embed(query), dtype=np.float64)
        nq = np.linalg.norm(q)
        if nq == 0:
            q = np.zeros_like(q)
        else:
            q = q / nq
        sims = self._matrix @ q
        order = np.argsort(-sims)
        out = []
        for idx in order[:k]:
            s = float(sims[idx])
            if s > 0:
                out.append((int(idx), s))
        return out

    @property
    def size(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[0])
