"""Слоты диалога и небольшие типы-модели (pydantic)."""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SlotNames:
    PROJECT_ID = "project_id"
    TARIFF = "tariff"
    PAYMENT_VALID_UNTIL = "payment_valid_until"
    CITY_TIMEZONE = "city_timezone"
    EMAIL_NOTIFICATION = "email_notification_enabled"
    AUTO_SCALE_CHANGE_ALLOWED = "auto_scale_change_allowed"


class SlotUpdate(BaseModel):
    """Дельта слотов за один ход (structured output): что удалось узнать."""

    project_id: Optional[str] = Field(None, description="Идентификатор проекта, например PRJ-35")
    tariff: Optional[str] = Field(None, description="Текущий тариф клиента: Start / Pro / Enterprise")
    payment_valid_until: Optional[str] = Field(None, description="Месяц, до которого оплачен тариф, формат YYYY-MM")
    city_timezone: Optional[str] = Field(None, description="Город или часовой пояс клиента")
    email_notification_enabled: Optional[bool] = Field(None, description="Дублировать ли ответы клиенту на email")
    auto_scale_change_allowed: Optional[bool] = Field(
        None, description="Разрешает ли клиент менять настройки автоскейлинга"
    )


class SlotPolicy(BaseModel):
    """Что делает агент с каждым слотом: нужен ли слот для задачи или для профиля."""

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


class DialogTurn(BaseModel):
    """Снимок одного хода диалога для журнала бюджета."""

    turn: int
    history_tokens_est: int
    actual_input_tokens: Optional[int] = None
    actual_output_tokens: Optional[int] = None
    action: Literal["none", "summarize+trim", "trim_only", "stop"]
    dropped: int = 0


TURN_SEPARATOR = "\n<<SUMMARY_TRUNCATION>>\n"
