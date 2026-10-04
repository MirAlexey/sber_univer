"""Критерий 2: бюджет контекста - фактические токены и суммаризация."""

from langchain_core.messages import AIMessage, HumanMessage

from src.context import RollingSummaryPipeline, actual_usage_tokens


def test_actual_usage_tokens_from_response():
    ai = AIMessage(content="ok")
    ai.usage_metadata = {"input_tokens": 4990, "output_tokens": 828}
    inp, out = actual_usage_tokens(ai)
    assert isinstance(inp, int)
    assert isinstance(out, int)
    assert inp == ai.usage_metadata["input_tokens"]
    assert out == ai.usage_metadata["output_tokens"]
def _fat_history(n_msgs):
    pad = "слово повторяется многажды " * 690
    return [HumanMessage(content=f"M{i}: " + pad) for i in range(n_msgs)]


def test_trimming_shrinks_history_and_adds_summary_marker():
    history = _fat_history(26)
    pipe = RollingSummaryPipeline(budget_tokens=6000)
    window, summary = pipe.run(history)

    assert len(window) < len(history)
    assert summary  # вытесненная часть стала свёрткой
    assert pipe.log
    assert window[0].content.startswith("<<SUMMARY>>")


def test_untouched_history_when_under_budget():
    history = [HumanMessage(content="короткая реплика")]
    pipe = RollingSummaryPipeline(budget_tokens=6500)
    window, summary = pipe.run(history)
    assert len(window) == len(history)
    assert summary == ""
