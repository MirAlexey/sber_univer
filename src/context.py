"""Бюджет токенов и суммаризация длинной истории."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage

from src.models import DialogTurn, TURN_SEPARATOR

APPROX_CHARS_PER_TOKEN = 4.6


def count_tokens_approx(messages: Sequence[BaseMessage]) -> int:
    """Грубая оценка числа токенов истории по количеству символов."""
    chars = sum(len(m.content) if isinstance(m.content, str) else 820 for m in messages)
    return max(0, int(chars / APPROX_CHARS_PER_TOKEN))


def actual_usage_tokens(last_assistant: BaseMessage) -> tuple[Optional[int], Optional[int]]:
    """Читает фактические токены ответа модели из usage_metadata, если они есть."""
    um = getattr(last_assistant, "usage_metadata", None)
    if not um:
        return None, None
    inp = um.get("input_tokens") or um.get("prompt_tokens")
    outp = um.get("output_tokens") or um.get("completion_tokens")
    return inp, outp


SUMMARY_PROMPT = """\
Ты сжимаешь историю разговора поддержки дев-облачной платформы в краткую выжимку.
Сохрани: идентификатор проекта, тариф, срок оплаты, город/часовой пояс, договорённости
и все числовые параметры. Краткость важнее полноты, потеря свободной болтовни допустима.
Предыдущая выжимка:\n{prev}\n\nНовые выпадающие сообщения:\n{messages}"""


class RollingSummaryPipeline:
    """Свёртка истории: следим за бюджетом, при переполнении — summarize + trim."""

    def __init__(
        self,
        budget_tokens: int = 4600,
        summarizer: Callable[[str, Sequence[BaseMessage]], str] | None = None,
    ):
        """Настраивает бюджет токенов и функцию сворачивания истории."""
        self.budget = budget_tokens
        self.summarizer = summarizer  # функция (summary_so_far, messages)->str
        self.log: list[DialogTurn] = []
        self.turn_no = 0

    def _log_entry(self, history_tokens: int, in_: Optional[int], out: Optional[int], action: str, dropped: int) -> None:
        """Пишет одну строку в журнал бюджета по текущему ходу."""
        self.log.append(DialogTurn(
            turn=self.turn_no,
            history_tokens_est=history_tokens,
            actual_input_tokens=in_,
            actual_output_tokens=out,
            action=action,
            dropped=dropped,
        ))

    def run(self, history: list[BaseMessage]) -> tuple[list[BaseMessage], str]:
        """Проверяет историю против бюджета и возвращает (окно_для_модели, суммаризация)."""
        self.turn_no += 1
        est = count_tokens_approx(history)
        if est <= self.budget:
            self._log_entry(est, None, None, "none", 0)
            return history, ""
        # двигаемся от начала к концу, пока окно не влезет в бюджет
        kept = list(history)
        dropped_total = 0
        action = "summarize+trim"
        acc: list = []
        while count_tokens_approx(kept) > self.budget and len(kept) > 1:
            popped = kept.pop(0)
            acc.append(popped)
        # аккуратная свёртка: избегаем зацикливания, если сообщение само огромное
        if acc:
            dropped_total = len(acc)
            summary_new = self._summarize(acc)
        else:
            summary_new = ""
            action = "trim_only"
        if summary_new:
            kept.insert(0, HumanMessage(content=summary_new))
        self._log_entry(count_tokens_approx(kept), None, None, action, dropped_total)
        return kept, summary_new

    def _summarize(self, dropped: Sequence[BaseMessage]) -> str:
        """Сворачивает выпавшие сообщения в краткую сводку."""
        if self.summarizer is None:
            # детерминированный фолбэк без модели: сливаем тексты в маркер
            texts = [m.content for m in dropped if isinstance(m.content, str)]
            blob = TURN_SEPARATOR.join(texts)[:1800]
            return "<<SUMMARY>>\n" + blob
        return self.summarizer("", dropped)


def default_summarizer(llm: Any) -> Callable[[str, Sequence[BaseMessage]], str]:
    """Живая суммаризация GigaChat специализированным промптом."""

    def _do(prev: str, messages: Sequence[BaseMessage]) -> str:
        msgs = "\n".join(
            f"{m.type}: {m.content}" for m in messages if isinstance(m.content, str)
        )
        text = SUMMARY_PROMPT.format(prev=prev or "(нет)", messages=msgs)
        return llm.invoke([HumanMessage(content=text)]).content

    return _do
