#!/usr/bin/env python3
"""Критерий 2: наглядный разбор бюджета токенов и суммаризации на длинном диалоге.

Берёт сгенерированный длинный диалог (33 реплики пользователя), прогоняет его
через RollingSummaryPipeline с бюджетом = доля от полной истории и показывает,
как вытесняются самые ранние сообщения и какие ключевые факты сохранились в сводке.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage  # noqa: E402

from scenarios.dialogs import DIALOGS, ESSENTIALS, user_turn_count  # noqa: E402
from scenarios.params import ParamAccumulator  # noqa: E402
from src.context import RollingSummaryPipeline, count_tokens_approx  # noqa: E402


def main(argv=None):
    """Прогоняет длинный диалог через бюджет и показывает сжатие истории."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dialog", choices=list(DIALOGS), default="prj35")
    ap.add_argument("--ratio", type=float, default=0.55)
    ap.add_argument("--window", action="store_true")
    args = ap.parse_args(argv)

    meta = DIALOGS[args.dialog]
    msgs = [HumanMessage(content=m["content"]) for m in meta["messages"] if isinstance(m["content"], str)]
    total_est = count_tokens_approx(msgs)
    budget = int(total_est * args.ratio)

    print(f"Диалог: {meta['title']} | реплик пользователя: {user_turn_count(meta['messages'])}")
    print(f"Полная история: ~{total_est} токенов | бюджет окна: ~{budget} токенов (ratio={args.ratio})")
    print()

    acc = ParamAccumulator()
    pipe = RollingSummaryPipeline(budget_tokens=budget, summarizer=acc.summarizer)
    history = []
    fires = []
    for i, m in enumerate(msgs):
        history.append(m)
        window, _ = pipe.run(history)
        entry = pipe.log[-1]
        if entry.action != "none":
            fires.append((i, entry))

    if fires:
        print("=== Срабатывания сжатия (в хронологическом порядке) ===")
        for i, e in fires:
            print(f"  после реплики #{i:>2}: {e.action:<14} вытеснено: {e.dropped:>2}, оценка окна ~{e.history_tokens_est}")
    else:
        print("Сжатие не потребовалось: история уложилась в бюджет.")

    final_est = _safe_window_est(pipe, msgs)
    saving = (total_est - final_est) * 100.0 / max(total_est, 1)
    print()
    print(f"Оценка конечного окна: ~{final_est} токенов | экономия: {saving:.1f}% "
          f"| событий сжатия: {len(fires)}")
    print()
    print("=== Сводка ключевых фактов (офлайн-суммаризатор) ===")
    print(acc.block())

    lost = [f for f in ESSENTIALS[args.dialog] if not acc.has(f)]
    print()
    if lost:
        print("Потери ключевых фактов:", lost)
        return 1
    print("Ключевые факты сохранены:", ", ".join(ESSENTIALS[args.dialog]))
    return 0


def _safe_window_est(pipe, msgs):
    """Оценка токенов окна после полного прогона."""
    w, _ = pipe.run(msgs)
    return count_tokens_approx(w)


if __name__ == "__main__":
    sys.exit(main())
