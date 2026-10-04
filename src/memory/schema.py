"""Схема памяти о пользователе: типы записей, горизонты хранения, доверие.

Критерий 5 задания 2: явная схема записи и разные горизонты хранения для
разных видов сведений. Здесь вид сведений задаётся kind, срок жизни -
RETENTION_DAYS[kind], а источник описывает доверие (TRUST_ORDER).
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any


class MemoryKind(str, enum.Enum):
    FACT = "fact"                 # стабильные свойства пользователя (город, тариф)
    PRODUCT = "product"           # сведения о продуктах/проектах клиента
    PREFERENCE = "preference"     # предпочтения коммуникации (email, push)
    EPISODE = "episode"           # события (инциденты, обращения)
    PROFILE_SUMMARY = "summary"   # агрегированный профиль (продукт консолидации)


# Горизонт хранения по виду сведений. None = хранить бессрочно.
RETENTION_DAYS: dict[MemoryKind, int | None] = {
    MemoryKind.FACT: 730,
    MemoryKind.PRODUCT: 540,
    MemoryKind.PREFERENCE: 720,
    MemoryKind.EPISODE: 545,
    MemoryKind.PROFILE_SUMMARY: None,
}

HORIZON_LABELS: dict[MemoryKind, str] = {
    MemoryKind.FACT: "2 года",
    MemoryKind.PRODUCT: "~1.5 года",
    MemoryKind.PREFERENCE: "2 года",
    MemoryKind.EPISODE: "~1.5 года",
    MemoryKind.PROFILE_SUMMARY: "бессрочно",
}


# Источники: чем ниже — тем менее надёжно. Правила разрешения противоречий
# опираются на эту иерархию.
class Source:
    USER_INFERRED = "user_inferred"   # модель домыслила из реплик (самое низкое доверие)
    ASSISTANT = "assistant"           # помощник записал после уточнения
    USER = "user"                     # пользователь сказал прямо
    VERIFIED = "verified"             # проверено интеграцией/документом


TRUST_ORDER: dict[str, int] = {
    Source.USER_INFERRED: 1,
    Source.ASSISTANT: 2,
    Source.USER: 3,
    Source.VERIFIED: 4,
}

SRC_LABEL = {
    Source.USER_INFERRED: "домысел",
    Source.ASSISTANT: "ассистент",
    Source.USER: "пользователь",
    Source.VERIFIED: "проверено",
}


@dataclass
class MemoryRecord:
    client_id: str
    kind: MemoryKind
    attr: str                            # ключ записи, напр. "tariff"
    value: Any                           # JSON-совместимое значение
    source: str = Source.USER
    confidence: float = 1.0
    observed_at: str = ""                # ISO date
    expires_at: str | None = None        # ISO date или None
    tombstoned: bool = False             # мягкое удаление
    masked: bool = False                 # значение стёрто, но структурная запись есть
    mandatory: bool = False              # обязана храниться (напр. финансы, комплаенс)
    ticket_id: str | None = None         # связь с обращением
    note: str = ""

    def age_days(self, today: date) -> int:
        return (today - date.fromisoformat(self.observed_at)).days if self.observed_at else 0

    def expired(self, today: date) -> bool:
        return bool(self.expires_at) and date.fromisoformat(self.expires_at) < today

    def applicable(self, today: date) -> bool:
        return not self.tombstoned and not self.expired(today)


def make_expiry(kind: MemoryKind, observed: date) -> str | None:
    days = RETENTION_DAYS.get(kind)
    return (observed + timedelta(days=days)).isoformat() if days else None


@dataclass
class AuditEntry:
    ts: str
    op: str                             # PUT_NEW / REFRESH / UPDATE / REJECT / MASK / PURGE
    client_id: str
    kind: str
    attr: str
    actor: str = "agent"
    reason: str = ""
    extra: dict = field(default_factory=dict)
