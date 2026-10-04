"""Фоновая переработка памяти вне диалога (sleep-time consolidation).

Критерий 6: дорогая переработка (дедупликация, агрегация в профиль-сводку)
выносится за пределы диалогового контура. Здесь это отдельно скомпилированный
LangGraph-граф, который запускается скриптом/планировщиком, а не при ответе.
"""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from src.memory.schema import MemoryKind
from src.memory.store import MemoryEngine


class ConsolidationState(TypedDict, total=False):
    client_id: str
    facts: list[dict]      # [{"attr","value","observed_at"}]
    merged: int
    summary: str
    errors: list[str]


def run_consolidation(engine: MemoryEngine, client_id: str, llm=None) -> dict:
    """Запуск фоновой консолидации: объединение дублей и построение сводки.

    llm — опциональная чат-модель для генерации краткой сводки; без неё
    сводка собирается детерминированно из ключей/значений.
    """
    app = _build_consolidation_graph(engine, llm)
    return app.invoke({"client_id": client_id})


def _build_consolidation_graph(engine: MemoryEngine, llm=None):
    def node_collect(state: ConsolidationState) -> dict:
        client_id = state["client_id"]
        facts = [
            {
                "kind": r.kind.value,
                "attr": r.attr,
                "value": r.value,
                "observed_at": r.observed_at,
                "source": r.source,
            }
            for r in engine.records(client_id)
            if r.kind in (MemoryKind.FACT, MemoryKind.PRODUCT, MemoryKind.PREFERENCE)
        ]
        return {"facts": facts}

    def node_merge(state: ConsolidationState) -> dict:
        # Простейшая дедупликация по ключу attr с сохранением самой свежей/доверенной.
        best: dict[str, dict] = {}
        for fact in state["facts"]:
            key = f"{fact['kind']}/{fact['attr']}"
            cur = best.get(key)
            if cur is None:
                best[key] = fact
                continue
            if (fact.get("observed_at", "") > cur.get("observed_at", "")
                    or (fact["observed_at"] == cur["observed_at"] and _trust_weight(fact) > _trust_weight(cur))):
                best[key] = fact
        merged = len(state["facts"]) - len(best)
        return {"merged": merged, "facts": list(best.values())}

    def node_summarize(state: ConsolidationState) -> dict:
        if not state["facts"]:
            return {"summary": ""}
        bullets = "\n".join(
            f"- {f['kind']}/{f['attr']} = {f['value']} ({f['observed_at']})"
            for f in state["facts"]
        )
        if llm is None:
            return {"summary": bullets}
        prompt = (
            "Ты консолидируешь профиль пользователя поддержки. Сократи следующие "
            "факты до плотной сводки из 3-5 строк без потери чисел и дат:\n" + bullets
        )
        try:
            return {"summary": llm.invoke(prompt).content}
        except Exception:
            return {"summary": ""}

    def node_stats(state: ConsolidationState) -> dict:
        agg = _aggregate_summary(state["client_id"], state["summary"])
        if agg:
            engine.write(
                client_id=state["client_id"],
                kind=MemoryKind.PROFILE_SUMMARY,
                attr="_consolidated_profile",
                value=agg,
                source="verified",
                confidence=0.995,
                actor="consolidation_job",
                reason="фоновая консолидация (sleep-time)",
                mandatory=True,
            )
        return {}

    g = StateGraph(ConsolidationState)
    g.add_node("collect", node_collect)
    g.add_node("merge_duplicates", node_merge)
    g.add_node("build_summary", node_summarize)
    g.add_node("stats", node_stats)
    g.add_edge(START, "collect")
    g.add_edge("collect", "merge_duplicates")
    g.add_edge("merge_duplicates", "build_summary")
    g.add_edge("build_summary", "stats")
    g.add_edge("stats", END)
    return g.compile()


def _trust_weight(fact: dict) -> int:
    from src.memory.schema import TRUST_ORDER

    return TRUST_ORDER.get(fact.get("source", ""), 0)


def _aggregate_summary(client_id: str, summary: str) -> dict:
    return {"client_id": client_id, "summary": summary}
