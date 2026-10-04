"""Критерий 3: гибридный поиск — лексика + семантика, слияние, точные id."""

import src.kb.builtin_docs as S  # noqa: F401
from src.kb.builtin_docs import DEMO_QUERY_NUM, DEMO_QUERY_TARIF_LIMIT


def test_exact_identifier_beats_semantics(offline_env):
    _, kb, _ = offline_env
    # "06-Т" почти не имеет смысла для эмбеддера, но находится лексически
    hits = kb.search(DEMO_QUERY_NUM, as_of="2026-06-10", k=3)
    assert hits
    assert hits[0].doc.doc_num == "06-Т"


def test_blending_lex_and_dense(offline_env):
    _, kb, _ = offline_env
    stage = kb.describe_stage(DEMO_QUERY_TARIF_LIMIT, as_of="2026-06-10")
    assert stage["lex_top"], "ожидали лексическую выдачу"
    assert stage["dense_top"], "ожидали семантическую выдачу"
    assert stage["rrf_top"], "слияние не дало результата"
    # цель в топе RRF
    tops = [num for num, _ in stage["rrf_top"]]
    assert "06-Т" in tops


def test_both_channels_visited_by_two_queries(offline_env):
    _, kb, _ = offline_env
    st_num = kb.describe_stage(DEMO_QUERY_NUM, as_of="2026-06-10")
    st_sem = kb.describe_stage(DEMO_QUERY_TARIF_LIMIT, as_of="2026-06-10")
    # лексика важна для id, семантика для смыслового запроса
    assert st_num["lex_top"][0][0] == "06-Т"
    assert st_sem["dense_top"], "смысловой запрос обязан находиться и на эмбеддингах"


def test_passages_carry_period_labels(offline_env):
    _, kb, _ = offline_env
    ps = kb.passages("лимит стартового тарифа", as_of="2026-06-10", k=6)
    assert ps
    assert any(pt.startswith("[06-Т v2") for pt in ps)
    for pt in ps:
        head, _, _ = pt.partition("]")
        assert head.startswith("[") and " v" in head
