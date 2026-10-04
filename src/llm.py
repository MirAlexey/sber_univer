"""Интерфейсы модели: чат-модель и эмбеддер.

Основной путь — langchain-gigachat напрямую (как в ноутбуках занятий 6-7):
ChatGigaChat + GigaChatEmbeddings. Библиотека импортируется лениво, чтобы тесты
без сети и без ключей могли работать на дублях из tests/fakes.py.
"""

from __future__ import annotations

import hashlib
from typing import Protocol

from src.models import SlotUpdate, slot_policies


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
        from langchain_gigachat.chat_models import ChatGigaChat
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

class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...

    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Определённый офлайн-эмбеддер для тестов и каркаса.

    Хэширует признаки слов в плотный вектор (Bag-of-Words + sign hashing +
    TF-нормировка). Смысловая близость тут грубая, зато поведение
    детерминировано и не требует сети.
    """

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

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


def get_embeddings(force_hash: bool = False):
    """Эмбеддер: живой GigaChatEmbeddings или офлайн HashEmbedder.

    hash_now указывается, когда эмбеддер должен работать без сети
    (тесты/скринкаст без ключей).
    """
    import os
    if force_hash or not os.environ.get("GIGACHAT_CREDENTIALS"):
        return HashEmbedder()
    try:
        from langchain_gigachat import GigaChatEmbeddings
    except ImportError:  # pragma: no cover
        return HashEmbedder()
    return GigaChatEmbeddings(
        model=os.environ.get("EMBED_MODEL", "EmbeddingsGigaChat"),
        **_gigachat_env(),
    )


# ---------------------------------------------------------------------------
# Промпты, общие для узлов графа
# ---------------------------------------------------------------------------

SYSTEM_GUIDE = """\
You are a customer support assistant for the DevCloud development platform.

Rules:
- Collect the user's requisites progressively from the conversation. New
  information replaces old guesses; conflicting statements are resolved towards
  the most recent user utterance.
- Before answering a factual question, you MUST call search_knowledge and answer
  STRICTLY from its returned passages.
- Every factual answer MUST end with inline source references in square
  brackets using the returned form, e.g. [06-Т v2].
- If the retrieved passage covers a different time period than the user asks
  about, say so and pick the passage whose period fits the requested moment.
- Small talk is fine, but never invent numbers or conditions from memory.
"""


def extractor_fields_description() -> str:
    items = [f"- {pol.name}: {pol.label}" for pol in slot_policies().values()]
    return "Known slots to update this turn:\n" + "\n".join(items)


def profile_block_label(kind: str) -> str:
    return f"known user facts ({kind}):"
