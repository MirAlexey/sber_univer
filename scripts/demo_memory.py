#!/usr/bin/env python3
"""Демо памяти: сессии, конфликт, удаление, аудит, консолидация."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Критерии 5-6: долговременная память по образцу сценариев dialogs.jsonl.

Три "сессии" клиента: ранняя говорит факты, средняя обновляет тариф,
поздняя опирается на старое. Потом запрос на удаление и консолидация.
Работает офлайн (HashEmbedder), без ключей.
"""

from datetime import date

from src.llm import HashEmbedder
from src.memory.consolidation import run_consolidation
from src.memory.schema import MemoryKind, Source
from src.memory.store import MemoryEngine


def main():
    """Демонстрирует память: сессии, конфликт, удаление, аудит, консолидацию."""
    embedder = HashEmbedder()
    engine = MemoryEngine(embedder=embedder, clock=lambda: date.fromisoformat("2026-09-01"))

    print("--- сессия 1: клиент сообщает факты")
    engine.write("client-011", MemoryKind.PRODUCT, "projects", "PRJ-98", source=Source.USER, actor="sev")
    engine.write("client-011", MemoryKind.FACT, "tariff", "Pro", source=Source.USER, actor="sev")
    engine.write("client-011", MemoryKind.FACT, "city_timezone", "Novosibirsk", source=Source.USER, actor="sev")

    print("--- сессия 2: тариф сменился на Start, доверие такое же -> UPDATE")
    engine.write("client-011", MemoryKind.FACT, "tariff", "Start", source=Source.USER, actor="sev")

    print("--- сессия 3: агент достаёт профиль по запросу")
    for rec, score in engine.search("client-011", "какой тариф и город", top_k=3):
        print("  recalled:", rec.attr, "=", rec.value, f"(rel={score:.2f})")

    print("--- запрос клиента на удаление; платежи обязаны храниться")
    engine.write("client-011", MemoryKind.PRODUCT, "payments", "paid-till-june-2027",
                 source=Source.VERIFIED, mandatory=True, actor="billing")
    report = engine.delete("client-011", kinds=[MemoryKind.FACT, MemoryKind.PRODUCT], reason="удаление по запросу")
    print("  purged:", report["purged"])
    print("  masked(kept):", report["kept_mandatory"])

    print("--- фоновая консолидация вне диалога")
    run_consolidation(engine, "client-011")
    summ = engine.get("client-011", MemoryKind.PROFILE_SUMMARY, "_consolidated_profile")
    print("  summary-record:", bool(summ), "| masked-kept:", bool(summ.masked if summ else False))

    print("--- журнал аудита по клиенту")
    for entry in engine.audit_report("client-011"):
        print("  ", entry.op, entry.kind, entry.attr, "-", entry.reason)


if __name__ == "__main__":
    main()
