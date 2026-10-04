#!/usr/bin/env bash
#
# Воспроизводимый запуск одной командой (задание 2).
#   bash run.sh               -> создать venv, установить зависимости, прогнать тесты
#   bash run.sh test          -> только тесты
#   bash run.sh demo-budget   -> демо критерия 2 (бюджет/суммаризация)
#   bash run.sh demo-memory   -> демо критериев 5-6 (память/аудит/удаление)
#   bash run.sh demo-temporal -> демо критерия 4 (редакции по датам)
#   bash run.sh run           -> живой диалог (нужны GIGACHAT_CREDENTIALS в .env)
#
set -euo pipefail
cd "$(dirname "$0")"
MODE="${1:-test}"

INSTALL_CMD=""
VENV_PY="python3"
if command -v uv >/dev/null 2>&1; then
  INSTALL_CMD="uv sync --group dev --extra gigachat"
  VENV_PY="uv run python"
  echo "[run.sh] используем uv"
else
  if [ ! -d ".venv" ]; then
    echo "[run.sh] создаю .venv"
    python3 -m venv .venv
  fi
  VENV_PY=".venv/bin/python"
  INSTALL_CMD="$VENV_PY -m pip install --quiet \
      langgraph langchain langchain-core pydantic numpy python-dotenv \
      langchain-gigachat pytest pytest-asyncio"
fi

if [ "$MODE" = "install" ]; then
  eval "$INSTALL_CMD"
  echo "[run.sh] установка завершена"
  exit 0
fi

eval "$INSTALL_CMD"

if [ ! -f ".env" ] && [ "$MODE" != "test" ]; then
  echo "[run.sh] нет .env (демо офлайн работают и без него; для живого режима скопируйте .env.example и впишите GIGACHAT_CREDENTIALS)"
fi

case "$MODE" in
  test)
    exec $VENV_PY -m pytest tests -q ;;
  demo-budget)
    exec $VENV_PY scripts/demo_budget.py --dialog "${2:-prj35}" ;;
  demo-memory)
    exec $VENV_PY scripts/demo_memory.py ;;
  demo-temporal)
    exec $VENV_PY scripts/demo_temporal.py ;;
  run)
    exec $VENV_PY main.py ;;
  *)
    echo "неизвестный режим: $MODE"; exit 2 ;;
esac
