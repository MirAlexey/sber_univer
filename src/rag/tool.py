"""Инструмент агента для поиска по базе знаний."""

from __future__ import annotations

from datetime import date
from typing import Optional

from langchain_core.tools import tool

from src.config import SETTINGS, deps
from src.rag.service import HybridSearch


def _as_of(arg: Optional[str]) -> str:
    if arg and arg.strip():
        return arg.strip()
    if SETTINGS.default_as_of:
        return SETTINGS.default_as_of
    return date.today().isoformat()


@tool
async def search_knowledge(query: str, as_of_date: Optional[str] = None) -> str:
    """Поиск по базе знаний DevCloud: верни пассажи, релевантные `query`.

    Возвращает до четырёх пассажей с номером документа, версией редакции и
    периодом действия. Факты приводи точно по тексту пассажа и добавляй тег в
    квадратных скобках (например [06-Т v2]). Параметр `as_of_date` (YYYY-MM-DD)
    передавай, когда пользователь спрашивает про конкретный момент в прошлом.
    """
    svc: HybridSearch | None = deps.kb
    if svc is None:
        return "База знаний пока недоступна."
    date_str = _as_of(as_of_date)
    passages = svc.passages(query, as_of=date_str, k=4)
    if not passages:
        return "По этому запросу и дате документы не найдены."
    return "\n\n".join(passages)
