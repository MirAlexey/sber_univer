"""Интерактивный диалог с агентом задания 2.

Запуск: cp .env.example .env (вписать GIGACHAT_CREDENTIALS), затем make run.
Без ключей можно крутить демо-скрипты (make demo-temporal / demo-memory) —
они работают офлайн на встроенных данных.
"""

import asyncio
import json
import os
import sys
import uuid

from dotenv import load_dotenv

load_dotenv()

THREAD = uuid.uuid4().hex[:12]
USER_ID = os.environ.get("DEMO_USER_ID", "client-001")


def _pretty(obj: dict) -> str:
    """Аккуратно печатает словарь для артефакта сессии."""
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


def _is_transient(exc: Exception) -> bool:
    """Транзиентные сбои API (лимит запросов, таймауты) — стоит повторить ход."""
    name = type(exc).__name__
    if "RateLimit" in name or "Timeout" in name or "Connection" in name:
        return True
    return any(code in str(exc) for code in ("429", "500", "502", "503", "504"))


def main() -> None:
    from src.graph.build import bootstrap, run_config
    from src.graph.nodes import export_snapshot

    try:
        graph = bootstrap(require_llm=True)
    except (ValueError, RuntimeError) as exc:
        print(f"[ошибка запуска] {exc}")
        print("Впишите GIGACHAT_CREDENTIALS в .env или посмотрите офлайн-демо: make demo-temporal / demo-memory")
        sys.exit(1)

    # Один event loop на всю сессию: иначе асинхронные объекты GigaChat
    # привязываются к закрытому циклу и ломаются со 2-го хода.
    try:
        asyncio.run(_repl(graph, export_snapshot))
    except KeyboardInterrupt:
        pass


async def _repl(graph, export_snapshot) -> None:
    from src.graph.build import run_config

    config = run_config(THREAD, user_id=USER_ID)
    print("Агент задания 2 (DevCloud support). Поможет по документам БЗ со ссылкой на источник.")
    print("Примеры: 'какой лимит воркеров у Start', 'что делать при QUOTA_EXCEEDED', exit — выход.\n")

    snapshot = {}
    while True:
        try:
            user_input = input("Вы: ")
        except EOFError:
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break

        result = None
        attempts = 0
        while True:
            try:
                result = await graph.ainvoke(
                    {"messages": [{"role": "user", "content": user_input}]}, config=config
                )
                break
            except KeyboardInterrupt:
                break
            except Exception as exc:  # noqa: BLE001
                attempts += 1
                delay = 2 ** attempts  # 2, 4, 8 секунд
                if _is_transient(exc) and attempts <= 3:
                    print(f"[перебор] {type(exc).__name__}; ждём {delay}с и пробуем снова…")
                    await asyncio.sleep(delay)
                    continue
                print(f"\n[ошибка графа] {type(exc).__name__}: {exc}")
                print("Продолжаем диалог.\n")
                break

        if result is None:
            continue

        answer = result["messages"][-1].content
        print(f"\nАгент: {answer}\n")
        snapshot = export_snapshot(result)

    print("\n=== Артефакт сессии ===")
    print(_pretty(snapshot))


if __name__ == "__main__":
    main()
