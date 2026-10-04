"""Критерий 4: отбор редакций по моменту времени. Один вопрос — разные ответы."""

from datetime import date

import pytest

import src.kb.seeds as S  # noqa: F401,F811
from src.kb.catalog import catalog_registry
from src.kb.seeds import AS_OF_DECEMBER, AS_OF_JUNE, DEMO_QUERY_TARIF_LIMIT


@pytest.fixture
def catalog_fresh():
    return catalog_registry.get()


def test_december_vs_june_select_different_editions(offline_env):
    _, kb, _ = offline_env

    dec = kb.search(DEMO_QUERY_TARIF_LIMIT, as_of=AS_OF_DECEMBER, k=3)
    jun = kb.search(DEMO_QUERY_TARIF_LIMIT, as_of=AS_OF_JUNE, k=3)

    dec_start = next(h for h in dec if h.doc.doc_id == "doc-start-limits")
    jun_start = next(h for h in jun if h.doc.doc_id == "doc-start-limits")

    assert dec_start.edition.version == 1
    assert "до 5 параллельных воркеров" in dec_start.edition.text
    assert jun_start.edition.version == 2
    assert "до 10 параллельных воркеров" in jun_start.edition.text


def test_version_not_active_yet_is_absent(offline_env):
    _, kb, _ = offline_env
    hits = kb.search("Enterprise", as_of="2025-11-20", k=5)
    assert all(h.doc.doc_id != "doc-ent-limits" for h in hits)


def test_edition_on_boundaries(catalog_fresh):
    doc = next(d for d in catalog_fresh.docs if d.doc_id == "doc-start-limits")
    assert doc.edition_on(date.fromisoformat("2026-01-14")).version == 1
    assert doc.edition_on(date.fromisoformat("2026-01-15")).version == 2
    assert doc.edition_on(date.fromisoformat("2026-12-31")).version == 2


def test_api_rate_editions_over_time(offline_env):
    _, kb, _ = offline_env
    dec = kb.search("лимиты API запросов у Pro", as_of="2025-12-01", k=4)
    jun = kb.search("лимиты API запросов у Pro", as_of="2026-06-10", k=4)
    d1 = next(h for h in dec if h.doc.doc_id == "doc-api-rates")
    d2 = next(h for h in jun if h.doc.doc_id == "doc-api-rates")
    assert d1.edition.version == 1
    assert "600" in d1.edition.text
    assert d2.edition.version == 2
    assert "900" in d2.edition.text
