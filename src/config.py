"""Настройки проекта и общий реестр зависимостей."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    agent_model: str = os.environ.get("AGENT_MODEL", "GigaChat-Pro")
    embed_model: str = os.environ.get("EMBED_MODEL", "EmbeddingsGigaChat")
    default_as_of: str = os.environ.get("DEFAULT_AS_OF", "")  # пусто -> сегодня
    memory_budget_tokens: int = int(os.environ.get("MEMORY_BUDGET_TOKENS", "700"))
    dialog_budget_tokens: int = int(os.environ.get("DIALOG_BUDGET_TOKENS", "3200"))
    recall_top_k: int = int(os.environ.get("RECALL_TOP_K", "4"))
    memory_file: str = os.environ.get("MEMORY_FILE", "")


SETTINGS = Settings()


@dataclass
class Deps:
    """Реестр живых объектов. Пустой означание того, что всё ещё не собрано."""

    llm: object = None            # ChatGigaChat (или дубль в тестах)
    extractor: object = None      # llm.with_structured_output(SlotUpdate)
    embedder: object = None       # протокол embed(text)->list[float]
    kb: object = None             # src.rag.service.HybridSearch
    memory: object = None         # src.memory.store.MemoryEngine
    store: object = None          # langgraph InMemoryStore (долговременная память)
    budget: object = None         # src.context.RollingSummaryPipeline

    def reset(self) -> None:
        """Обнуляет все поля реестра."""
        for name in list(self.__dict__.keys()):
            if name != "_lock":
                self.__dict__[name] = None


deps = Deps()


def configure(
    *,
    llm=None,
    extractor=None,
    embedder=None,
    kb=None,
    memory=None,
    store=None,
    budget=None,
) -> None:
    """Пересобирает зависимости. Вызывается один раз при старте приложения."""
    if llm is not None:
        deps.llm = llm
    if extractor is not None:
        deps.extractor = extractor
    if embedder is not None:
        deps.embedder = embedder
    if kb is not None:
        deps.kb = kb
    if memory is not None:
        deps.memory = memory
    if store is not None:
        deps.store = store
    if budget is not None:
        deps.budget = budget


def reset_defaults() -> None:
    """Возврат к пустому реестру (для тестов)."""
    deps.reset()
