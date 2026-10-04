"""Сборка диалогового графа (LangGraph)."""

from __future__ import annotations

from typing import Annotated, Any, Optional

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.memory import InMemoryStore
from typing_extensions import TypedDict

from src.config import deps, SETTINGS
from src.context import RollingSummaryPipeline, default_summarizer
from src.kb.catalog import catalog_registry
from src.llm import build_extractor, get_chat_llm, get_embeddings
from src.memory.store import MemoryEngine
from src.rag.service import HybridSearch
from src.graph.nodes import (
    build_profile,
    extract_slots,
    manage_context,
    persist_user,
    respond,
)

STEP_LIMIT_PARENT = 962  # щедрый верхний лимит шагов графа (сам цикл инструментов ограничен внутри узла)


class AgentState(TypedDict):
    """Состояние диалогового графа: сообщения, слоты, память, журнал бюджета."""

    messages: Annotated[list[BaseMessage], add_messages]
    slots: dict
    client_id: Optional[str]
    profile_block: str
    budget_log: list[dict]
    citations: list[str]
    sync_state: str


def build_graph(**overrides) -> Any:
    """Собирает диалоговый граф; параметры override нужны тестам, чтобы подставить дубли.

    Если в overrides чего-то нет, берутся уже сконфигурированные объекты deps,
    либо создаются офлайн-заменители (HashEmbedder, встроенные документы БЗ, MemoryEngine).
    """
    _ensure_offline_stack(**overrides)

    g = StateGraph(AgentState)
    g.add_node("extract_slots", extract_slots)
    g.add_node("manage_context", manage_context)
    g.add_node("build_profile", build_profile)
    g.add_node("respond", respond)
    g.add_node("persist_user", persist_user)

    g.add_edge(START, "extract_slots")
    g.add_edge("extract_slots", "manage_context")
    g.add_edge("manage_context", "build_profile")
    g.add_edge("build_profile", "respond")
    g.add_edge("respond", "persist_user")
    g.add_edge("persist_user", END)

    graph = g.compile(checkpointer=MemorySaver(), store=deps.store)
    return graph


def _ensure_offline_stack(**overrides) -> None:
    """Достраивает минимально необходимые объекты, если их ещё нет."""
    if overrides.get("llm") is not None:
        deps.llm = overrides["llm"]
    if overrides.get("extractor") is not None:
        deps.extractor = overrides["extractor"]

    if deps.embedder is None:
        deps.embedder = get_embeddings(force_hash=(deps.llm is None))
    if deps.kb is None:
        deps.kb = HybridSearch(catalog_registry.get(), deps.embedder)
    if deps.memory is None:
        deps.memory = MemoryEngine(
            embedder=deps.embedder,
            snapshot_path=SETTINGS.memory_file or None,
        )
    if deps.store is None:
        deps.store = InMemoryStore()
    if deps.budget is None:
        summarizer = default_summarizer(deps.llm) if deps.llm is not None else None
        deps.budget = RollingSummaryPipeline(
            budget_tokens=SETTINGS.dialog_budget_tokens,
            summarizer=summarizer,
        )


def bootstrap(require_llm: bool = True) -> Any:
    """Живой запуск: GigaChat + GigaChatEmbeddings; без ключей кидает ValueError."""
    if require_llm:
        llm = get_chat_llm()
        deps.llm = llm
        deps.extractor = build_extractor(llm)
    return build_graph()


def run_config(thread_id: str, user_id: str = "anonymous", callbacks: Optional[list] = None) -> dict:
    """Конфигурация одного запуска графа: сессия (thread_id) + идентичность пользователя."""
    cfg: dict = {
        "configurable": {"thread_id": thread_id, "user_id": user_id},
        "recursion_limit": STEP_LIMIT_PARENT,
    }
    if callbacks:
        cfg["callbacks"] = callbacks
    return cfg
