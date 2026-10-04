"""Дубли модели и экстрактора для офлайн-тестов (без сети, без ключей).

Повторяет идею tests/fakes.py шаблона задания 3, но под конкретику задания 2:
отдаём по очереди AIMessage с tool_call на поиск и финальный ответ со ссылкой.
"""

from __future__ import annotations

import uuid
from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


def call_tool(name: str, args: dict, **attrs) -> AIMessage:
    msg = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])
    for k, v in attrs.items():
        setattr(msg, k, v)
    return msg


def say(text: str, **attrs) -> AIMessage:
    msg = AIMessage(content=text)
    for k, v in attrs.items():
        setattr(msg, k, v)
    return msg


class ScriptedChatModel(BaseChatModel):
    """Дубль чат-модели: возвращает сообщения из script по очереди."""

    script: list[AIMessage]
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-task2"

    def bind_tools(self, tools, **kwargs):
        return self

    def bind_tools_return_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        if self.calls >= len(self.script):
            raise AssertionError("сценарий дубля закончился раньше, чем агент завершился")
        message = self.script[self.calls]
        self.calls += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def ainvoke(self, input: Any, config=None, **kwargs):
        return self._pick_next()

    def _pick_next(self):
        if self.calls >= len(self.script):
            raise AssertionError("сценарий дубля закончился раньше, чем агент завершился")
        message = self.script[self.calls]
        self.calls += 1
        return message


class FakeExtractor:
    """Дубль with_structured_output: возвращает заранее заданные SlotUpdate."""

    def __init__(self, updates: list | None = None):
        self.updates = list(updates or [])
        self.calls = 0

    def push(self, update) -> None:
        self.updates.append(update)

    async def ainvoke(self, messages: Sequence[BaseMessage], config=None, **kwargs):
        if self.calls >= len(self.updates):
            raise AssertionError("экстрактор слотов не имел очередного ответа")
        upd = self.updates[self.calls]
        self.calls += 1
        return upd

    async def arun(self, *args, **kwargs):
        return await self.ainvoke(args)
