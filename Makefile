install:
	uv sync --group dev --extra gigachat
test:
	uv run pytest
test-offline-fast:
	uv run pytest tests/ -x -q
run:
	uv run python main.py
demo-temporal:
	uv run python scripts/demo_temporal.py
demo-memory:
	uv run python scripts/demo_memory.py
consolidate:
	uv run python scripts/consolidate.py --client client-001
demo-budget:
	uv run python scripts/demo_budget.py --dialog prj35
