"""Поиск по эмбеддингам (плотный канал)."""

from __future__ import annotations

import numpy as np

from src.llm import GigaChatEmbedder, HashEmbedder


class DenseIndex:

    def __init__(self, embedder: HashEmbedder | GigaChatEmbedder):
        """Сохраняет эмбеддер; индекс строится при вызове fit."""
        self.embedder = embedder
        self._matrix: np.ndarray | None = None
        self._dim = 0

    def fit(self, texts: list[str]) -> None:
        """Векторизует тексты и складывает нормализованную матрицу."""
        vectors = [np.asarray(self.embedder.embed(t), dtype=np.float64) for t in texts]
        matrix = np.stack(vectors) if vectors else np.zeros((0, 764), dtype=np.float64)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        matrix = matrix / norms
        self._matrix = matrix
        self._dim = int(matrix.shape[1])

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        """Косинусный поиск: топ-k позиций с положительной близостью."""
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
        """Число документов в индексе."""
        return 0 if self._matrix is None else int(self._matrix.shape[0])
