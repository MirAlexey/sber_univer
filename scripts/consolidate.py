#!/usr/bin/env python3
"""Фоновая пересборка памяти клиента (вне диалога)."""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Фоновая переработка памяти вне диалога (кандидат в scheduler/CI)."""

import argparse

from src.llm import HashEmbedder
from src.memory.consolidation import run_consolidation
from src.memory.store import MemoryEngine


def main():
    """Запускает фоновую консолидацию памяти для клиента."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--client", default="client-001")
    args = parser.parse_args()

    engine = MemoryEngine(embedder=HashEmbedder())
    before = len(engine.all_keys())
    outcome = run_consolidation(engine, args.client)
    after = len(engine.all_keys())
    print(f"client={args.client} fact_rows_seen={len(outcome.get('facts', []))} "
          f"dup_merged={outcome.get('merged', 0)} records {before}->{after}")


if __name__ == "__main__":
    main()
