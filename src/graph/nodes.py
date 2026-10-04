"""Узлы диалога: слоты, бюджет, память, ответ со ссылкой на источник."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from src.config import SETTINGS, deps
from src.context import RollingSummaryPipeline, count_tokens_approx
from src.llm import SYSTEM_GUIDE
from src.memory.schema import MemoryKind, Source
from src.models import slot_policies
from src.rag.tool import search_knowledge

if TYPE_CHECKING:
    from src.graph.build import AgentState

MAX_ANSWER_STEPS = 4


def get_client_id(config: RunnableConfig) -> str:
    """Идентификатор пользователя из конфигурации запуска."""
    return (config or {}).get("configurable", {}).get("user_id", "anonymous")


def _last_human_content(messages) -> str:
    """Текст последнего сообщения пользователя."""
    for m in reversed(messages):
        if m.type == "human":
            return m.content if isinstance(m.content, str) else ""
    return ""


# ---------------------------------------------------------------------------
# 1. structured output: накопление слотов
# ---------------------------------------------------------------------------

async def extract_slots(state: "AgentState") -> dict:
    """Узнает новые слоты из последней реплики и сливает их в состояние."""
    extractor = deps.extractor
    if extractor is None:
        raise RuntimeError("extractor is not configured (проверьте bootstrap)")
    last_user = _last_human_content(state["messages"])
    upd = await extractor.ainvoke([
        SystemMessage(
            content=(
                "Верни только те слоты, которые узнал из ЭТОГО сообщения пользователя; "
                "незатронутые поля должны быть null. Известные слоты: "
                + ", ".join(pol.name for pol in slot_policies().values())
            )
        ),
        HumanMessage(content=last_user),
    ])
    merged = dict(state.get("slots") or {})
    for name, val in upd.model_dump(exclude_none=True).items():
        merged[name] = val
    return {"slots": merged}


# ---------------------------------------------------------------------------
# 2. бюджет контекста: фактические токены + суммаризация вытесненного
# ---------------------------------------------------------------------------

async def manage_context(state: "AgentState") -> dict:
    """Проверяет бюджет токенов и при переполнении сворачивает старую историю."""
    history = state.get("messages") or []
    pipeline: RollingSummaryPipeline | None = deps.budget
    if pipeline is None:
        pipeline = RollingSummaryPipeline(budget_tokens=SETTINGS.dialog_budget_tokens)
        deps.budget = pipeline
    # Прогоняем историю через конвейер бюджета; он вернет ужатое окно.
    window, _ = pipeline.run(history)   # summary уже внесён маркером внутрь window
    logs = list(state.get("budget_log") or [])
    if pipeline.log:
        logs.append(pipeline.log[-1].model_dump())
    return {"budget_log": logs, "messages": window}


# ---------------------------------------------------------------------------
# 3. долговременная память пользователя -> блок для промпта
# ---------------------------------------------------------------------------

async def build_profile(state: "AgentState", config: RunnableConfig) -> dict:
    """Достает из памяти факты о пользователе для системного промпта."""
    memory = deps.memory
    if memory is None:
        return {"profile_block": ""}
    client_id = state.get("client_id") or get_client_id(config)
    recalls = memory.search(client_id, _last_human_content(state["messages"]), top_k=SETTINGS.recall_top_k)
    # Отбираем факты, пока они укладываются в токенный бюджет памяти.
    lines = []
    spent = 0
    for rec, score in recalls:
        text = f"- {rec.kind.value}/{rec.attr} = {rec.value} (до {rec.expires_at or '∞'}, rel={score:.2f})"
        cost = count_tokens_approx([SystemMessage(content=text)])
        if spent + cost > SETTINGS.memory_budget_tokens:
            break
        lines.append(text)
        spent += cost
    return {"profile_block": "\n".join(lines)}


# ---------------------------------------------------------------------------
# 4. ответ со ссылкой на источник: LLM + инструмент гибридного поиска
# ---------------------------------------------------------------------------

async def respond(state: "AgentState") -> dict:
    """Зовет модель с инструментом поиска, исполняет вызовы инструментов и возвращает ответ."""
    llm = deps.llm
    if llm is None:
        raise RuntimeError("llm is not configured (проверьте bootstrap)")
    if deps.kb is None:
        raise RuntimeError("knowledge base is not configured")

    profile = state.get("profile_block") or ""
    summary = state.get("summary") or ""
    # Собираем системный блок: правила + профиль пользователя + свёрнутая история.
    sys_blocks = [SYSTEM_GUIDE]
    if profile:
        sys_blocks.append("Долговременный профиль пользователя (используй, только если относится к делу):\n" + profile)
    if summary:
        sys_blocks.append("Ранее свёрнутая история:\n" + summary)
    system_prompt = "\n\n".join(sys_blocks)

    history = state.get("messages") or []
    chain = [SystemMessage(content=system_prompt), *history]
    model_with_tools = llm.bind_tools([search_knowledge])

    citations: list[str] = list(state.get("citations") or [])
    assistant_msg: AIMessage | None = None
    usage: tuple[Any, Any] = (None, None)

    # Цикл инструментов: пока модель предлагает вызовы, исполняем их.
    for _ in range(MAX_ANSWER_STEPS):
        response = await model_with_tools.ainvoke(chain)
        if not getattr(response, "tool_calls", None):
            assistant_msg = response
            usage = _usage_of(response)
            break
        chain.append(response)
        for tc in response.tool_calls:
            tool_fn = {"search_knowledge": search_knowledge}.get(tc.get("name"))
            if tool_fn is None:
                chain.append(
                    ToolMessage(
                        content=f"Unknown tool '{tc.get('name')}', skipped.",
                        tool_call_id=tc.get("id") or "",
                        name=tc.get("name") or "unknown",
                    )
                )
                continue
            try:
                out = await tool_fn.ainvoke(tc.get("args") or {})
            except Exception as exc:  # noqa: BLE001
                out = f"Tool error: {exc}"
            if isinstance(out, str):
                for tag in _tags_in(out):
                    if tag not in citations:
                        citations.append(tag)
            chain.append(
                ToolMessage(
                    content=str(out),
                    tool_call_id=tc.get("id") or "",
                    name=tc.get("name") or "unknown",
                )
            )

    if assistant_msg is None:
        assistant_msg = AIMessage(content="Step limit reached before finishing.")

    logs = list(state.get("budget_log") or [])
    in_tok, out_tok = usage
    # Пишем фактические токены ответа в журнал бюджета.
    if (in_tok is not None or out_tok is not None) and logs:
        last = dict(logs[-1])
        last["actual_input_tokens"] = in_tok
        last["actual_output_tokens"] = out_tok
        logs[-1] = last

    return {
        "messages": [assistant_msg],
        "citations": citations,
        "budget_log": logs,
    }


def _usage_of(response: AIMessage) -> tuple[Any, Any]:
    """Достает фактические токены ответа из usage_metadata."""
    um = getattr(response, "usage_metadata", None) or {}
    return um.get("input_tokens"), um.get("output_tokens")


def _tags_in(text: str) -> list[str]:
    """Ищет в тексте ссылки вида [NN-Б vN]."""
    tags: list[str] = []
    idx = text.find("[")
    while idx >= 0:
        end = text.find("]", idx)
        if end > idx:
            tag = text[idx + 1:end]
            if "," in tag:
                tags.append(tag)
        idx = text.find("[", idx + 1)
    return tags


# ---------------------------------------------------------------------------
# 5. перенос новых слотов в долговременную память + зеркало в LangGraph Store
# ---------------------------------------------------------------------------

async def persist_user(state: "AgentState", config: RunnableConfig) -> dict:
    """Кладет новые слоты в долговременную память и зеркалирует в Store."""
    memory = deps.memory
    store = deps.store
    client_id = state.get("client_id") or get_client_id(config)
    slots = state.get("slots") or {}
    if memory is not None:
        _persist_to_engine(memory, client_id, slots)
    if store is not None:
        store.put(("user_profile", client_id), "slots_v1", {"slots": slots, "synced": True})
    return {"sync_state": "done"}


def _persist_to_engine(memory, client_id: str, slots: dict) -> None:
    """Пишет профильные слоты в движок памяти."""
    mapping = {
        "project_id": (MemoryKind.PRODUCT, "projects", str),
        "tariff": (MemoryKind.FACT, "tariff", str),
        "payment_valid_until": (MemoryKind.PRODUCT, "payments", str),
        "city_timezone": (MemoryKind.FACT, "city_timezone", str),
        "email_notification_enabled": (MemoryKind.PREFERENCE, "comm_email", bool),
    }
    for slot_name, val in slots.items():
        if val is None or slot_name not in mapping:
            continue
        kind, attr, conv = mapping[slot_name]
        memory.write(
            client_id=client_id,
            kind=kind,
            attr=attr,
            value=conv(val),
            source=Source.USER,
            confidence=0.980,
            actor="dialog_slot",
            reason=f"слот {slot_name} из диалога",
        )


def export_snapshot(state: "AgentState") -> dict:
    """Артефакт запуска: слоты, ссылки, журнал бюджета."""
    """Артефакт запуска для скринкаста и README."""
    return {
        "slots": state.get("slots"),
        "citations": state.get("citations"),
        "budget_log": state.get("budget_log"),
        "profile_block": state.get("profile_block"),
    }
