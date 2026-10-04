"""Каталог документов базы знаний: редакции и сроки их действия."""

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
        """Сохраняет список семейств документов."""
        self.docs = docs
        # версии индексов строятся лениво, см. src/rag/service.py

    def all_active_texts(self, d: date) -> list[tuple[KnowledgeDoc, Edition]]:
        """Пары (документ, редакция), действующие на дату d."""
        out = []
        for doc in self.docs:
            ed = doc.edition_on(d)
            if ed is not None:
                out.append((doc, ed))
        return out


_seed_registry: list[KnowledgeDoc] = []


def register_seed(doc: KnowledgeDoc) -> KnowledgeDoc:
    """Регистрирует семейство документов во встроенном каталоге."""
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
        """Создает реестр без активного каталога."""
        self._catalog = None

    def set(self, catalog: Catalog) -> None:
        """Устанавливает активный каталог (для тестов)."""
        with _LOCK:
            self._catalog = catalog

    def get(self) -> Catalog:
        """Возвращает активный каталог, при необходимости создает его из семян."""
        with _LOCK:
            if self._catalog is None:
                self._catalog = default_catalog()
            return self._catalog


catalog_registry = CatalogRegistry()
