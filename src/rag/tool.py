"""Инструмент поиска по базе знаний, который агент вызывает сам.

Это не фиксированная цепочка, а @tool: наличие-отсутствие вызова решает сама
модель в процессе диалога (критерий 3). Даты "на момент" берутся из аргумента
либо из активной даты по умолчанию (критерий 4).
"""

from __future__ import annotations

from datetime import date
from typing import Optional

from langchain_core.tools import tool

from src.config import SETTINGS, deps
from src.models import CitedSource
from src.rag.service import HybridSearch


def _as_of(arg: Optional[str]) -> str:
    if arg and arg.strip():
        return arg.strip()
    if SETTINGS.default_as_of:
        return SETTINGS.default_as_of
    return date.today().isoformat()


@tool
async def search_knowledge(query: str, as_of_date: Optional[str] = None) -> str:
    """Search the DevCloud knowledge base for passages relevant to `query`.

    Returns up to four passages tagged with the document number, edition version
    and validity period. Facts must be reported exactly as stated in a returned
    passage, with the bracketed tag appended (e.g. [06-Т v2]). Pass
    `as_of_date` (YYYY-MM-DD) when the user asks about a specific past moment.
    """
    svc: HybridSearch | None = deps.kb
    if svc is None:
        return "Knowledge base is not ready."
    date_str = _as_of(as_of_date)
    passages = svc.passages(query, as_of=date_str, k=4)
    if not passages:
        return "No documents found for this query and date."
    return "\n\n".join(passages)


def hits_to_citations(hits) -> list[CitedSource]:
    return [
        CitedSource(
            doc_id=h.doc.doc_id,
            doc_num=h.doc.doc_num,
            version=h.edition.version,
            valid_from=h.edition.valid_from,
            valid_to=h.edition.valid_to,
            title=h.edition.title or h.doc.family_title,
        )
        for h in hits
    ]


def citation_tags(passages: list[str]) -> list[str]:
    """Вытаскивает [NN-Б vN]-теги из текста пассажей (для контроля ссылки)."""
    out = []
    for p in passages:
        start = p.find("[")
        end = p.find("]", start)
        if start >= 0 and end > start:
            out.append(p[start + 1:end])
    return out
