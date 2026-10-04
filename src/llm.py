"""Интерфейсы модели: чат-модель и эмбеддер.

Основной путь — langchain-gigachat напрямую (как в ноутбуках занятий 6-7):
GigaChat + GigaChatEmbeddings. Библиотека импортируется лениво, чтобы тесты
без сети и без ключей могли работать на дублях из tests/fakes.py.
"""

from __future__ import annotations

import hashlib

from src.models import SlotUpdate


# ---------------------------------------------------------------------------
# Чат-модель
# ---------------------------------------------------------------------------

def _gigachat_env() -> dict:
    """Ключи GigaChat берутся из окружения (формат langchain-gigachat)."""
    import os
    return {
        "credentials": os.environ.get("GIGACHAT_CREDENTIALS", ""),
        "scope": os.environ.get("GIGACHAT_SCOPE", "GIGACHAT_API_B2B"),
        "verify_ssl_certs": os.environ.get("GIGACHAT_VERIFY_SSL_CERTS", "False").lower() == "true",
    }


def get_chat_llm(**overrides):
    """Живой ChatGigaChat. TypeError без установленного пакета или без ключа."""
    try:
        from langchain_gigachat import GigaChat as ChatGigaChat
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Установите langchain-gigachat (uv sync включает только обязательное)") from exc
    cfg = {**_gigachat_env(), "temperature": 0.0, **overrides}
    cfg["model"] = overrides.get("model", __import__("src.config", fromlist=["SETTINGS"]).SETTINGS.agent_model)
    if not cfg["credentials"]:
        raise ValueError("GIGACHAT_CREDENTIALS не заданы (см. .env.example)")
    return ChatGigaChat(**cfg)


def build_extractor(llm):
    """Структурный выход для слотов (structured output)."""
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
        if not self.norm:
            return v
        mag = sum(x * x for x in v) ** 0.5
        if mag < 1e-12:
            return v
        return [x / mag for x in v]

    def embed(self, text: str) -> list[float]:
        return self._normalize(self._featurize(text))


class GigaChatEmbedder:
    """Живой эмбеддер: оборачивает GigaChatEmbeddings под единый метод embed()."""

    def __init__(self, raw):
        self._raw = raw

    def embed(self, text: str) -> list[float]:
        return [float(x) for x in self._raw.embed_query(text)]


def get_embeddings(force_hash: bool = False):
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
