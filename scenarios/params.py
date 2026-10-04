"""Офлайн-замена суммаризатора: выхватывает ключевые факты предметной области.

Имитирует "специализированный под задачу промпт": вместо LLM извлекает из
выпадающих сообщений структурные элементы (проекты, тарифы, города, месяцы,
числа с единицами) и ведёт аккумулятор уникальных находок. Этого достаточно,
чтобы проверять в тестах и демо, что при сжатии истории суть не теряется.
"""

from __future__ import annotations

import re

_PROJECT = re.compile(r"PRJ-\d+")
_TARIFF = re.compile(r"\b(Start|Pro|Enterprise)\b")
_CITY = re.compile(r"\b(Сочи|Владивосток|Новосибирск|Москва|Санкт-Петербург)\b")
_MONTH = re.compile(r"\b(январ|феврал|март|апрел|ма[йя]|июн|июл|август|сентябр|октябр|ноябр|декабр)[а-я]*")
_NUM_UNIT = re.compile(r"\b(\d{1,3})\s+(?:[^\s.0-9]+\s+){0,2}(воркер\w*|дней?|месяц\w*|раз в год|%)")
_TIMESTAMP = re.compile(r"\b(\d{2}:\d{2})\b")
_KEYWORD = re.compile(r"\b(burst|email|push|SMTP|SPF|возврат|перенос|автоскейлинг|лим[ии]т)\w*", re.I)


def harvest(text: str) -> list[str]:
    """Вылавливает из текста структурные факты: проекты, тарифы, даты, числа."""
    found: list[str] = []
    found += _PROJECT.findall(text)
    found += _TARIFF.findall(text)
    found += _CITY.findall(text)
    found += [m.group(0) for m in _MONTH.finditer(text)]
    found += [f"{m.group(1)} {m.group(2)}" for m in _NUM_UNIT.finditer(text)]
    found += [m.group(0) for m in _TIMESTAMP.finditer(text)]
    found += [m.group(1).capitalize() for m in _KEYWORD.finditer(text)]
    return found


def normalize(token: str) -> str:
    """Приводит факт к нижнему регистру без ё."""
    return token.lower().replace("ё", "е").strip()


class ParamAccumulator:
    """Копит уникальные факты по мере появления и выдаёт компактную сводку."""

    def __init__(self) -> None:
        self._uniq: set[str] = set()

    def absorb(self, text: str) -> None:
        """Добавляет факты из текста в набор."""
        for tok in harvest(text):
            self._uniq.add(normalize(tok))

    def has(self, token: str) -> bool:
        """Есть ли похожий факт в наборе."""
        needle = normalize(token)
        for item in self._uniq:
            if needle in item or item in needle:
                return True
        return False

    def block(self) -> str:
        """Сводка всех собранных фактов."""
        head = "СВОДКА КЛЮЧЕВЫХ ФАКТОВ (офлайн-суммаризатор):"
        if not self._uniq:
            return head + " (пусто)"
        body = "; ".join(sorted(self._uniq))
        return head + "\n" + body

    def summarizer(self, prev: str, messages) -> str:
        """Адаптер для RollingSummaryPipeline: поглощает сообщения и возвращает сводку."""
        """Интерфейс для RollingSummaryPipeline: (предыдущая сводка, выпавшие сообщения) -> текст."""
        for m in messages:
            content = m.content if isinstance(m.content, str) else ""
            self.absorb(content)
        if prev:
            self.absorb(prev)
        return self.block()
