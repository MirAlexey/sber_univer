"""Типовые модели состояния диалога и слотов.

Слоты — структурированный набор реквизитов, который агент заполняет по ходу
длинного диалога (structured output). Часть слотов относится к "задаче"
(реквизиты для поиска), часть — к долговременному профилю пользователя и
уходит в слой памяти (src/memory/schema.py).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# Слоты, которые агент обязан постепенно собрать из диалога.
class SlotNames:
    PROJECT_ID = "project_id"
    TARIFF = "tariff"
    PAYMENT_VALID_UNTIL = "payment_valid_until"
    CITY_TIMEZONE = "city_timezone"
    EMAIL_NOTIFICATION = "email_notification_enabled"
    AUTO_SCALE_CHANGE_ALLOWED = "auto_scale_change_allowed"


ALL_SLOTS: tuple[str, ...] = (
    SlotNames.PROJECT_ID,
    SlotNames.TARIFF,
    SlotNames.PAYMENT_VALID_UNTIL,
    SlotNames.CITY_TIMEZONE,
    SlotNames.EMAIL_NOTIFICATION,
    SlotNames.AUTO_SCALE_CHANGE_ALLOWED,
)


class SlotUpdate(BaseModel):
    """Дель по одному вызову structured output: какие слоты удалось узнать."""

    project_id: Optional[str] = Field(None, description="Идентификатор проекта, например PRJ-35")
    tariff: Optional[str] = Field(None, description="Текущий тариф клиента: Start / Pro / Enterprise")
    payment_valid_until: Optional[str] = Field(None, description="Месяц, до которого оплачен тариф, формат YYYY-MM")
    city_timezone: Optional[str] = Field(None, description="Город или часовой пояс клиента")
    email_notification_enabled: Optional[bool] = Field(None, description="Хочет ли клиент дублировать ответы на email")
    auto_scale_change_allowed: Optional[bool] = Field(
        None, description="Разрешает ли клиент менять настройки автоскейлинга"
    )
    # NULL в any field = слот не узнан в этом ходе.


SLOT_FIELD_LABELS: dict[str, str] = {
    SlotNames.PROJECT_ID: "проект",
    SlotNames.TARIFF: "тариф",
    SlotNames.PAYMENT_VALID_UNTIL: "оплата до",
    SlotNames.CITY_TIMEZONE: "город/часовой пояс",
    SlotNames.EMAIL_NOTIFICATION: "дублирование на email",
    SlotNames.AUTO_SCALE_CHANGE_ALLOWED: "разрешена смена автоскейлинга",
}


class SlotPolicy(BaseModel):
    """Что делает агент с каждым слотом: требуется для задачи или для профиля."""

    name: str
    label: str
    persists_to_profile: bool = False


def slot_policies() -> dict[str, SlotPolicy]:
    """Схема записи слотов: какие реквизиты считаются профильными (уходят в память)."""
    return {
        SlotNames.PROJECT_ID: SlotPolicy(name=SlotNames.PROJECT_ID, label="проект", persists_to_profile=True),
        SlotNames.TARIFF: SlotPolicy(name=SlotNames.TARIFF, label="тариф", persists_to_profile=True),
        SlotNames.PAYMENT_VALID_UNTIL: SlotPolicy(
            name=SlotNames.PAYMENT_VALID_UNTIL, label="оплата до", persists_to_profile=True
        ),
        SlotNames.CITY_TIMEZONE: SlotPolicy(
            name=SlotNames.CITY_TIMEZONE, label="город/часовой пояс", persists_to_profile=True
        ),
        SlotNames.EMAIL_NOTIFICATION: SlotPolicy(
            name=SlotNames.EMAIL_NOTIFICATION, label="email-дублирование", persists_to_profile=True
        ),
        SlotNames.AUTO_SCALE_CHANGE_ALLOWED: SlotPolicy(
            name=SlotNames.AUTO_SCALE_CHANGE_ALLOWED, label="автоскейлинг", persists_to_profile=False
        ),
    }


class CitedSource(BaseModel):
    """Одна ссылка на найденный источник, которую обязан привести агент."""

    doc_id: str = Field(description="Стабильный идентификатор документа")
    doc_num: str = Field(description="Служебный номер, например 06-Т")
    version: int = Field(description="Номер редакции документа")
    valid_from: str = Field(description="Начало периода действия редакции")
    valid_to: Optional[str] = Field(None, description="Конец периода действия редакции или пусто")
    title: str = Field(description="Заголовок документа")


class DialogTurn(BaseModel):
    """Снимок одного хода диалога для журнала бюджета."""

    turn: int
    history_tokens_est: int
    actual_input_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    action: Literal["none", "summarize+trim", "trim_only", "stop"]
    dropped: int = 0


TURN_SEPARATOR = "\n<<SUMMARY_TRUNCATION>>\n"
