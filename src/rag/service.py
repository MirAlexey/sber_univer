"""Гибридный поиск: BM25 + эмбеддинги, слияние и реранк."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from src.kb.catalog import Catalog, Edition, KnowledgeDoc
from src.llm import GigaChatEmbedder, HashEmbedder
from src.rag.dense import DenseIndex
from src.rag.fusion import get_reranker, reciprocal_rank_fusion
from src.rag.lexical import BM25Index, boost_exact_identifiers


@dataclass(frozen=True)
class Hit:
    """Один найденный документ вместе с редакцией и оценкой."""

    doc: KnowledgeDoc
    edition: Edition
    score: float
    sources: tuple[str, ...]

    def passage_text(self, limit: int = 520) -> str:
        """Текст пассажа со служебным номером и периодом действия."""
        text = self.edition.text
        if len(text) > limit:
            cut = text[:limit].rsplit(" ", 1)[0] + "…"
            text = cut
        period_end = self.edition.valid_to or "∞"
        return (
            f"[{self.doc.doc_num} v{self.edition.version}, "
            f"{self.edition.valid_from}..{period_end}] "
            f"{self.edition.title or self.doc.family_title}: {text}"
        )

    def citation_tag(self) -> str:
        """Ссылка вида [Номер vВерсия]."""
        return f"[{self.doc.doc_num} v{self.edition.version}]"


@dataclass
class _Snapshot:
    as_of: date
    nums: list[str]                       # параллельный массив служебных номеров
    seq_for_row: list[tuple[KnowledgeDoc, Edition]]
    lex_docs: list[str]                   # склеенный текст для BM25
    bm25: Optional[BM25Index] = None
    dense: Optional[DenseIndex] = None


class HybridSearch:
    """Инструмент агента: лексика + семантика + RRF + реранк по состоянию на дату."""

    def __init__(self, catalog: Catalog, embedder: HashEmbedder | GigaChatEmbedder, prefer_cross_encoder: bool = True):
        """Сохраняет каталог, эмбеддер и настраивает реранкер."""
        self.catalog = catalog
        self.embedder = embedder
        self._reranker = get_reranker(prefer_cross_encoder)
        self._snapshots: dict[date, _Snapshot] = {}

    # ---- подготовка снимков -------------------------------------------------
    def _snapshot(self, as_of: date) -> _Snapshot:
        """Строит и кэширует индексы (лексика + плотные) для конкретной даты."""
        sn = self._snapshots.get(as_of)
        if sn is not None:
            return sn
        active = self.catalog.all_active_texts(as_of)
        rows_seq = [(doc, ed) for doc, ed in active]
        nums = [doc.doc_num for doc, _ in rows_seq]
        lex_docs = [self._composite(doc, ed) for doc, ed in rows_seq]
        sn = _Snapshot(
            as_of=as_of,
            nums=nums,
            seq_for_row=rows_seq,
            lex_docs=lex_docs,
            bm25=BM25Index(lex_docs),
            dense=DenseIndex(self.embedder),
        )
        sn.dense.fit(lex_docs)
        self._snapshots[as_of] = sn
        return sn

    @staticmethod
    def _composite(doc: KnowledgeDoc, ed: Edition) -> str:
        """Склеивает текст документа для индексации."""
        return f"{doc.doc_num}. {doc.family_title}. {ed.title}. {ed.section}. {ed.text}"

    # ---- поиск --------------------------------------------------------------
    def search(self, query: str, as_of: date | str, k: int = 4) -> list[Hit]:
        """Ищет топ-k документов на дату, склеивая лексику и семантику."""
        as_of = _to_date(as_of)
        sn = self._snapshot(as_of)
        # Лексический и семантический каналы.
        lex_run = [i for i, _ in boost_exact_identifiers(sn.bm25.search(query, k=10), sn.nums, query)]
        dense_run = [i for i, _ in sn.dense.search(query, k=10)]
        # Сливаем выдачи через RRF и берем топ.
        fused = reciprocal_rank_fusion([lex_run, dense_run], k=620)
        top = fused[:k]
        results: list[Hit] = []
        for idx, score in top:
            doc, ed = sn.seq_for_row[idx]
            sr = ["lex"] if idx in lex_run else []
            if idx in dense_run:
                sr.append("dense")
            results.append(Hit(doc=doc, edition=ed, score=score, sources=tuple(sr) or ("rrf",)))
        # последняя ступень: реранкер переставляет уже сформированные хиты
        results = self._final_rerank(results, query)
        return results[:k]

    def _final_rerank(self, hits: list[Hit], query: str) -> list[Hit]:
        """При наличии реранкера переставляет хиты по его оценкам."""
        if not getattr(self._reranker, "available", False):
            return hits
        scores = self._reranker.rerank(query, [(h.edition.text, query) for h in hits])
        return [h for h, _ in sorted(zip(hits, scores), key=lambda pr: -pr[1])]

    def passages(self, query: str, as_of: date | str, k: int = 4) -> list[str]:
        """Готовые тексты пассажей для модели."""
        return [h.passage_text() for h in self.search(query, as_of, k=k)]

    # ---- диагностика (для тестов и скринкаста) --------------------------------
    def describe_stage(self, query: str, as_of: date | str) -> dict:
        """Диагностика: что нашлось на каждом этапе поиска."""
        as_of = _to_date(as_of)
        sn = self._snapshot(as_of)
        lex_run = boost_exact_identifiers(sn.bm25.search(query, k=10), sn.nums, query)
        dense_run = sn.dense.search(query, k=10)
        fused = reciprocal_rank_fusion([[i for i, _ in lex_run], [i for i, _ in dense_run]], k=620)
        return {
            "as_of": as_of.isoformat(),
            "lex_top": [(sn.nums[i], round(s, 4)) for i, s in lex_run[:3]],
            "dense_top": [(sn.nums[i], round(s, 4)) for i, s in dense_run[:3]],
            "rrf_top": [(sn.nums[i], round(s, 4)) for i, s in fused[:4]],
        }


def _to_date(x: date | str) -> date:
    """Приводит дату (строка ISO или date) к типу date."""
    if isinstance(x, date):
        return x
    return datetime.strptime(x, "%Y-%m-%d").date()


