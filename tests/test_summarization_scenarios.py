"""Проверка критерия 2 на длинных диалогах: бюджет токенов и суммаризация.

Длинные диалоги лежат в scenarios/dialogs.py (по 33 реплики пользователя).
Тесты прогоняют их через RollingSummaryPipeline с малым бюджетом и проверяют,
что сжатие сработало, журнал бюджета ведётся и ключевые факты пережили сжатие.
"""

import pytest
from langchain_core.messages import HumanMessage

from src.context import RollingSummaryPipeline, count_tokens_approx
from scenarios.dialogs import DIALOGS, ESSENTIALS, user_turn_count
from scenarios.params import ParamAccumulator

KEYS = list(DIALOGS)


def _messages(key):
    return [HumanMessage(content=m["content"]) for m in DIALOGS[key]["messages"] if isinstance(m["content"], str)]


def _stream(key, budget=None):
    all_msgs = _messages(key)
    if budget is None:
        budget = int(count_tokens_approx(all_msgs) * 0.55)
    acc = ParamAccumulator()
    pipe = RollingSummaryPipeline(budget_tokens=budget, summarizer=acc.summarizer)
    history = []
    last_window = []
    for m in all_msgs:
        history.append(m)
        window, _ = pipe.run(history)
        last_window = window
    return pipe, acc, last_window


@pytest.mark.parametrize("key", KEYS)
def test_dialog_has_30_plus_user_turns(key):
    assert user_turn_count(DIALOGS[key]["messages"]) >= 30


@pytest.mark.parametrize("key", KEYS)
def test_summarization_actually_triggers(key):
    pipe, _, _ = _stream(key)
    actions = [e.action for e in pipe.log]
    assert any(a != "none" for a in actions)


@pytest.mark.parametrize("key", KEYS)
def test_budget_log_records_estimates(key):
    pipe, _, _ = _stream(key)
    assert pipe.log
    assert any(e.dropped > 0 for e in pipe.log)


@pytest.mark.parametrize("key", KEYS)
def test_key_facts_survive_compression(key):
    _, acc, _ = _stream(key)
    for fact in ESSENTIALS[key]:
        assert acc.has(fact), f"{key}: факт '{fact}' потерян при суммаризации"


@pytest.mark.parametrize("key", KEYS)
def test_first_message_leaves_live_window(key):
    _, _, last_window = _stream(key)
    joined = " ".join(m.content for m in last_window)
    snippet = _messages(key)[0].content[:20]
    assert snippet not in joined, "самое раннее сообщение осталось в окне без сжатия"
