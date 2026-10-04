import pytest
from datetime import date

from src.config import configure, reset_defaults


@pytest.fixture(autouse=True)
def _clean_deps():
    reset_defaults()
    yield
    reset_defaults()


@pytest.fixture
def offline_env():
    """Готовый офлайн-стек: HashEmbedder + каталог с встроенными документами + HybridSearch + MemoryEngine."""
    import src.kb.builtin_docs  # noqa: F401  (регистрация встроенных документов)
    from src.kb.catalog import catalog_registry
    from src.llm import HashEmbedder
    from src.memory.store import MemoryEngine
    from src.rag.service import HybridSearch

    embedder = HashEmbedder()
    kb = HybridSearch(catalog_registry.get(), embedder, prefer_cross_encoder=False)
    engine = MemoryEngine(embedder=embedder, clock=lambda: date(2026, 9, 1))
    configure(embedder=embedder, kb=kb, memory=engine)
    return embedder, kb, engine
