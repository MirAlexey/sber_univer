"""Критерий 1: диалоговый контур - слоты, checkpointer, Store, ссылка на источник.

Прогоняем граф без сети: дубль модели вызывает инструмент поиска и отвечает со
ссылкой, экстрактор слотов выдаёт заданные обновления. Проверяем накопление
слотов между ходами, межсессионную память и попадание фактических токенов в
журнал бюджета.
"""

from langchain_core.messages import HumanMessage
from src.memory.schema import MemoryKind
from src.models import SlotUpdate

from src.graph.build import build_graph, run_config
from src.config import deps
from tests.fakes import FakeExtractor, ScriptedChatModel, call_tool, say

SEARCH_ARGS = {"query": "какой лимит параллельных воркеров у Start?", "as_of_date": "2026-06-10"}
FINAL_TEXT = "Сейчас тариф Start позволяет до 10 параллельных воркеров (документ [06-Т v2])."


async def test_full_contour_single_thread(offline_env):
    _, _, engine = offline_env

    final = say(FINAL_TEXT, usage_metadata={"input_tokens": 493, "output_tokens": 617})
    llm = ScriptedChatModel(script=[call_tool("search_knowledge", SEARCH_ARGS), final, call_tool("search_knowledge", SEARCH_ARGS), final])
    extractor = FakeExtractor([
        SlotUpdate(project_id="PRJ-35"),
        SlotUpdate(tariff="Start", city_timezone="Europe/Moscow"),
    ])

    graph = build_graph(llm=llm, extractor=extractor)
    cfg = run_config("thread-A", user_id="client-777")

    first = await graph.ainvoke(
        {"messages": [HumanMessage(content="По проекту PRJ-35 воркеры падают с QUOTA_EXCEEDED")]}, config=cfg
    )
    assert first["slots"].get("project_id") == "PRJ-35"
    assert "06-Т v2" in first["messages"][-1].content
    assert any(c.startswith("06-Т v2") for c in first.get("citations", []))
    last_log = first["budget_log"][-1]
    assert last_log.get("actual_input_tokens") == final.usage_metadata["input_tokens"]
    assert last_log.get("actual_output_tokens") == final.usage_metadata["output_tokens"]

    second = await graph.ainvoke({"messages": [HumanMessage(content="Берите тариф Start")]}, config=cfg)
    slots = second["slots"]
    assert slots.get("project_id") == "PRJ-35"
    assert slots.get("tariff") == "Start"

    records = engine.records("client-777", MemoryKind.FACT)
    assert any(r.attr == "tariff" for r in records)

    item = deps.store.get(("user_profile", "client-777"), "slots_v1")
    assert item is not None
    assert item.value["slots"].get("project_id") == "PRJ-35"
