"""База знаний: документы с версиями/редакциями и их периодами действия.

Предметная область взята из сценариев dialogs.jsonl — поддержка дев-облачной
платформы: тарифы, лимиты воркеров, квоты, уведомления, автоскейлинг,
экспорт отчётов, возврат средств. Документы имеют редакции (version) и период
действия (valid_from..valid_to), поэтому один и тот же вопрос на разную дату
должен давать разные корректные ответы (критерий 4 задания 2).
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import date
from typing import Optional


@dataclass(frozen=True)
class Edition:
    """Одна редакция документа."""

    version: int
    valid_from: str  # ISO date, inclusive
    valid_to: Optional[str]  # ISO date inclusive or None = бессрочно
    text: str
    title: str = ""
    section: str = ""


@dataclass(frozen=True)
class KnowledgeDoc:
    doc_id: str  # стабильный идентификатор, например doc-start-limits
    doc_num: str  # служебный номер, например 06-Т (точный идентификатор для BM25)
    family_title: str
    editions: tuple[Edition, ...]

    def edition_on(self, d: date) -> Optional[Edition]:
        """Редакция, действующая на дату d. None, если документа тогда не было."""
        ord_ = d.toordinal()
        for ed in self.editions:
            lo = date.fromisoformat(ed.valid_from).toordinal()
            hi = (date.fromisoformat(ed.valid_to).toordinal() if ed.valid_to else 999_999)
            if lo <= ord_ <= hi:
                return ed
        return None


class Catalog:
    """Все семейства документов базы знаний плюс плоский список активных текстов."""

    def __init__(self, docs: list[KnowledgeDoc]):
        self.docs = docs
        # версии индексов строятся лениво, см. src/rag/service.py

    def all_active_texts(self, d: date) -> list[tuple[KnowledgeDoc, Edition]]:
        out = []
        for doc in self.docs:
            ed = doc.edition_on(d)
            if ed is not None:
                out.append((doc, ed))
        return out


_seed_registry: list[KnowledgeDoc] = []


def register_seed(doc: KnowledgeDoc) -> KnowledgeDoc:
    _seed_registry.append(doc)
    return doc


def default_catalog() -> Catalog:
    """Каталог из встроенных семян; потокобезопасно и идемпотентно."""
    if not _seed_registry:
        raise RuntimeError("seeds не загружены: импортируйте src.kb.seeds")
    return Catalog(list(_seed_registry))


_LOCK = threading.Lock()


class CatalogRegistry:
    """Один активный каталог, переустанавливаемый тестами (паттерн конфигурации)."""

    def __init__(self) -> None:
        self._catalog = None

    def set(self, catalog: Catalog) -> None:
        with _LOCK:
            self._catalog = catalog

    def get(self) -> Catalog:
        with _LOCK:
            if self._catalog is None:
                self._catalog = default_catalog()
            return self._catalog


catalog_registry = CatalogRegistry()
