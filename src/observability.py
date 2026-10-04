"""Минимум наблюдаемости для каркаса задания 2.

В отличие от задания 3 здесь не нужен тяжёлый контур трассировки: достаточно
журнала бюджета, ссылок на источники и лёгкого списка событий, который можно
показать в скринкасте. По желанию подключается Langfuse (как в шаблоне).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Tracer:
    events: list[dict] = field(default_factory=list)

    def emit(self, event: str, **details: Any) -> None:
        self.events.append({"event": event, **details})

    def reset(self) -> None:
        self.events.clear()

    def summary(self) -> list[dict]:
        return list(self.events)
