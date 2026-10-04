"""Подключение к GigaChat и эмбеддеры (живые и офлайн)."""

from __future__ import annotations

import hashlib
from typing import Any

from src.models import SlotUpdate


# ---------------------------------------------------------------------------
# Чат-модель
# ---------------------------------------------------------------------------

def _gigachat_env() -> dict:
    """Настройки подключения к GigaChat, собранные из переменных окружения."""
    """Ключи GigaChat берутся из окружения (формат langchain-gigachat)."""
    import os
    return {
        "credentials": os.environ.get("GIGACHAT_CREDENTIALS", ""),
        "scope": os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_B2B"),
        "verify_ssl_certs": os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "False").lower() == "true",
    }


def get_chat_llm(**overrides):
    """Настоящая GigaChat-модель для чата; ошибка, если пакет не установлен или нет ключа."""
    try:
        from langchain_gigachat import GigaChat as ChatGigaChat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Установите langchain-gigachat (uv sync включает только обязательное)") from exc
    cfg = {**_gigachat_env(), "temperature": 0.0, **overrides}
    cfg["model"] = overrides.get("model", __import__("src.config", fromlist=["SETTINGS"]).SETTINGS.agent_model)
    if not cfg["credentials"]:
        raise ValueError("GIGACHAT_CREDENTIALS не заданы (см. .env.example)")
    return ChatGigaChat(**cfg)


def build_extractor(llm: Any) -> object:
    """Возвращает LLM с фиксированной схемой слотов (structured output)."""
    return llm.with_structured_output(SlotUpdate)


# ---------------------------------------------------------------------------
# Эмбеддеры
# ---------------------------------------------------------------------------



class HashEmbedder:
    """Офлайн-эмбеддер (Bag-of-Words + sign hashing): для тестов и без сети."""

    def __init__(self, dim: int = 960, norm: bool = True):
        self.dim = dim
        self.norm = norm

    def _featurize(self, text: str) -> list[float]:
        """Сырой вектор признаков из слов текста."""
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        for tok, cnt in tf.items():
            h1 = int(hashlib.blake2b(tok.encode("utf-8"), digest_size=8).hexdigest(), 16)
            pos = h1 % self.dim
            sign = 1.0 if (h1 >> 511) & 1 else -1.0
            vec[pos] += sign * (1.0 + (cnt - 1) * 0.5)
        return vec

    def _normalize(self, v: list[float]) -> list[float]:
        """Нормализация вектора к единичной длине."""
        if not self.norm:
            return v
        mag = sum(x * x for x in v) ** 0.5
        if mag < 1e-12:
            return v
        return [x / mag for x in v]

    def embed(self, text: str) -> list[float]:
        """Эмбеддинг текста (офлайн-режим)."""
        return self._normalize(self._featurize(text))


class GigaChatEmbedder:
    """Живой эмбеддер: обёртка над GigaChatEmbeddings с кэшем векторов."""

    _CAP = 1024

    def __init__(self, raw):
        """Запоминает обёрнутый объект GigaChatEmbeddings и пустой кэш."""
        self._raw = raw
        self._cache = {}

    def embed(self, text: str) -> list[float]:
        """Эмбеддинг текста от GigaChat (повторные тексты берутся из кэша)."""
        hit = self._cache.get(text)
        if hit is not None:
            return hit
        vec = [float(x) for x in self._raw.embed_query(text)]
        if len(self._cache) >= self._CAP:
            self._cache.clear()
        self._cache[text] = vec
        return vec


def get_embeddings(force_hash: bool = False) -> HashEmbedder | GigaChatEmbedder:
    """Эмбеддер живого режима (GigaChat) или офлайн-заменитель (HashEmbedder).

    force_hash=True либо USE_HASH_EMBEDDER=1 возвращают офлайн-эмбеддер (тесты,
    демо без сети). В живом режиме имя модели опускается, если EMBED_MODEL пуст
    или равен "EmbeddingsGigaChat" — этот идентификатор сервер отклоняет (404).
    """
    import os
    if force_hash or os.environ.get("USE_HASH_EMBEDDER", "") == "1":
        return HashEmbedder()
    from langchain_gigachat import GigaChatEmbeddings
    cfg = _gigachat_env()
    model = os.environ.get("EMBED_MODEL", "")
    if model and model != "EmbeddingsGigaChat":
        cfg["model"] = model
    return GigaChatEmbedder(GigaChatEmbeddings(**cfg))


SYSTEM_GUIDE = """\
Ты — ассистент поддержки платформы разработки DevCloud.

Правила:
- Собирай реквизиты пользователя постепенно из диалога. Новые сведения
  заменяют прежние предположения; противоречия разрешай в пользу наиболее
  свежей реплики пользователя.
- Прежде чем отвечать на фактический вопрос, ОБЯЗАТЕЛЬНО вызови search_knowledge
  и отвечай строго по его результатам.
- Каждый фактический ответ ОБЯЗАН завершаться ссылкой на источник в квадратных
  скобках в форме из выдачи, например [06-Т v2].
- Если найденный пассаж покрывает другой период, чем спрашивает пользователь,
  скажи об этом и выбери пассаж, чей период соответствует нужному моменту.
- Короткая светская беседа допустима, но никогда не выдумывай цифры и условия
  по памяти.
"""
