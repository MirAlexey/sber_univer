"""Проверка устойчивости: строгий парсинг ссылок и мягкий фолбэк слотов."""

from langchain_core.messages import HumanMessage

from src.config import deps
from src.graph.nodes import _tags_in, extract_slots
from tests.fakes import FakeExtractor


def test_tags_in_ignores_fake_and_finds_real():
    real = "Смотрите документ [06-Т v2, 2026-01-15..∞], а это не ссылка [привет]."
    assert _tags_in(real) == ["06-Т v2"]
    assert _tags_in("тут нет ничего [см. выше] и [тест]") == []


async def test_extract_slots_graceful_on_bad_structured_output():
    class BoomExtractor(FakeExtractor):
        async def ainvoke(self, *args, **kwargs):
            raise ValueError("модель вернула кривой JSON")

    deps.extractor = BoomExtractor()
    state = {
        "messages": [HumanMessage(content="Мой тариф Start")],
        "slots": {"tariff": "Start"},
    }
    result = await extract_slots(state)
    assert result["slots"] == {"tariff": "Start"}
