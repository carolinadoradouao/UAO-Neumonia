install:
	uv sync

run:
	uv run python -m src.ui

check:
	uv run ruff check .
	uv run pytest