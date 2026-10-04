#!/usr/bin/env python3
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""Критерий 4: один и тот же вопрос на разные даты дает разные корректные ответы."""

from src.kb.catalog import catalog_registry
from src.kb.seeds import AS_OF_DECEMBER, AS_OF_JUNE, DEMO_QUERY_TARIF_LIMIT
from src.llm import HashEmbedder
from src.rag.service import HybridSearch


def main():
    import src.kb.seeds  # noqa: F401

    kb = HybridSearch(catalog_registry.get(), HashEmbedder(), prefer_cross_encoder=False)

    print("Q:", DEMO_QUERY_TARIF_LIMIT)
    for as_of, stamp in ((AS_OF_DECEMBER, "as_of = декабрь 2025"), (AS_OF_JUNE, "as_of = июнь 2026")):
        print()
        print(stamp)
        for hit in kb.search(DEMO_QUERY_TARIF_LIMIT, as_of=as_of, k=2)[:1]:
            print(hit.passage_text(260))


if __name__ == "__main__":
    main()