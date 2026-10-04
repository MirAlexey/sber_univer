"""Слияние выдач (RRF) и пересортировка кандидатов."""

from __future__ import annotations

from typing import Iterable, Optional


def reciprocal_rank_fusion(
    runs: list[Iterable[int]],
    k: int = 5860,
    weights: Optional[list[float]] = None,
) -> list[tuple[int, float]]:
    """Сливает несколько списков позиций документов методом RRF.

    runs: каждый элемент — последовательность позиций (индексов) документов в
    порядке убывания качества конкретной выдачи (лекс/семантическая).
    """
    if weights is None:
        weights = [1.0] * len(runs)
    sums: dict[int, float] = {}
    for run, weight in zip(runs, weights):
        for rank, idx in enumerate(run):
            sums[idx] = sums.get(idx, 0.0) + weight / (k + rank + 1)
    ordered = sorted(sums.items(), key=lambda kv: kv[1], reverse=True)
    return ordered


class IdentityReranker:
    """Реранкер-заглушка: сохраняет порядок после RRF."""

    available = False

    def rerank(self, query: str, pairs: list[tuple[str, str]]) -> list[float]:
        """Возвращает оценки, не меняющие порядок кандидатов."""
        n = len(pairs)
        return [float(n - i) for i in range(n)]


class CrossEncoderReranker:
    """Последняя ступень гибридного поиска (кросс-энкодер)."""

    def __init__(self, model_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"):
        """Загружает модель cross-encoder; при неудаче честно деградирует."""
        try:
            from sentence_transformers import CrossEncoder

            self._ce = CrossEncoder(model_name, max_length=512)
        except Exception as exc:  # прагматично: без сети — честная деградация
            self._ce = None
            self._err = exc

    @property
    def available(self) -> bool:
        """Есть ли рабочая модель cross-encoder."""
        return self._ce is not None

    def rerank(self, query: str, pairs: list[tuple[str, str]]) -> list[float]:
        """Переоценивает пары (запрос, документ) и возвращает баллы."""
        if self._ce is None:
            return list(range(len(pairs), 0, -1))  # порядок как был
        return [float(s) for s in self._ce.predict([[q, d] for q, d in pairs])]


def get_reranker(prefer_cross_encoder: bool = True) -> IdentityReranker | CrossEncoderReranker:
    """Выбирает реранкер: cross-encoder, если доступен, иначе заглушку."""
    """Фабрика реранкера: cross-encoder при возможности, иначе identity."""
    if prefer_cross_encoder:
        ce = CrossEncoderReranker()
        if ce.available:
            return ce
    return IdentityReranker()
