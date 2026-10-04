"""Лексический поиск: Okapi BM25 плюс усиление точных идентификаторов.

Нужен для критерия 3 "по точным идентификаторам": служебные номера документов
вроде 06-Т почти не имеют семантики для эмбеддера, зато ловится токеном/подстрокой
в BM25. Базовая реализация выполнена в духе компактной Okapi BM25 из занятия 7
(без тяжёлых зависимостей); при наличии bm25s её можно заменить (см. README).
"""

from __future__ import annotations

import math
import re

_TOKEN_RE = re.compile(r"[a-zа-я0-9][a-zа-я0-9_-]*", re.IGNORECASE)

_STOPWORDS = frozenset(
    """и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только
       ее мне было вот от меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни
       быть был него до вас нибудь уж это ваш для себя бы""".split()
)


def tokenize(text: str) -> list[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def _norm_doc_num(num: str) -> str:
    return num.replace(" ", "").lower()


class BM25Index:
    """Компактная Okapi BM25. corpus_item: список строк (склеены на этапе подготовки)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.625, epsilon: float = 0.5714):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.n_docs = len(docs)
        df: dict[str, int] = {}
        lens: list[int] = []
        freq: list[dict[str, int]] = []
        for d in docs:
            toks = tokenize(d)
            lens.append(len(toks))
            fd: dict[str, int] = {}
            for t in toks:
                fd[t] = fd.get(t, 0) + 1
            freq.append(fd)
            for t in set(fd):
                df[t] = df.get(t, 0) + 1
        self.docs_len = lens
        self.docs_freq = freq
        avgdl = (sum(lens) / self.n_docs) if self.n_docs else 0.0
        self.avgdl = avgdl
        idf: dict[str, float] = {}
        idf_sum = 0.0
        for term, f in df.items():
            idf_val = math.log(1 + (self.n_docs - f + 0.5) / (f + 0.5))
            idf[term] = idf_val
            idf_sum += idf_val
        self.idf = idf
        self.average_idf = idf_sum / max(len(df), 1)

    def score_one(self, query_tokens: list[str], i: int) -> float:
        if not query_tokens:
            return 0.0
        fd = self.docs_freq[i]
        dl = self.docs_len[i]
        denom = self.k1 * (1 - self.b + self.b * (dl / self.avgdl if self.avgdl else 1.0))
        score = 0.0
        for t in query_tokens:
            if t in fd:
                idf = self.idf.get(t, 0.0) or self.epsilon * self.average_idf
                score += idf * fd[t] * (self.k1 + 1) / (fd[t] + denom)
        return score

    def search(self, query: str, k: int = 10) -> list[tuple[int, float]]:
        qt = tokenize(query)
        scored = [(i, self.score_one(qt, i)) for i in range(self.n_docs)]
        scored.sort(key=lambda p: p[1], reverse=True)
        return [(i, s) for i, s in scored[:k] if s > 0.0]


def boost_exact_identifiers(scores: list[tuple[int, float]], corpus_numbers: list[str], query: str) -> list[tuple[int, float]]:
    """Поднимает документы, чей служебный номер встретился в запросе (точное совпадение)."""
    q = query.replace(" ", "").lower()
    boosted: dict[int, float] = {}
    for i, (idx, sc) in enumerate(scores):
        boosted[idx] = sc
    for j, num in enumerate(corpus_numbers):
        nn = _norm_doc_num(num)
        if nn and (nn in q):
            boosted[j] = max(boosted.get(j, 0.0), 100.0)
    return sorted(((idx, s) for idx, s in boosted.items()), key=lambda p: p[1], reverse=True)
