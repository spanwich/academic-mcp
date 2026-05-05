# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ MCP server for Zotero-backed academic paper access. Core code lives in `src/`: `server.py` and `server_http.py` are entry points, `config.py` loads settings, `models/` contains SQLAlchemy and ChromaDB wrappers, `processing/` handles PDF extraction and classification, and `zotero/` handles Zotero reads and sync. Operational scripts are at the repository root and in `scripts/`. Docker files live in `docker/`. Treat `data/`, `papers/`, and `logs/` as local artifacts.

## Build, Test, and Development Commands

Create and activate a virtual environment before local work:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Useful commands:

- `python test_setup.py` checks database, vectors, Ollama, embeddings, and Zotero setup.
- `./start_server.sh` starts the local MCP server and ensures Ollama is available.
- `./start_server_http.sh` starts the HTTP transport variant.
- `python zotero_import.py --collection "Collection Name"` imports one Zotero collection; use `--all` for everything.
- `pytest tests/` runs the test suite.
- `black src/` formats source files; `ruff check src/` runs lint checks.
- `docker compose -f docker/compose.yml build` builds the Docker stack.

## Coding Style & Naming Conventions

Use Black formatting with 88-character lines and Ruff for linting. Prefer typed Python interfaces; Mypy is configured with `disallow_untyped_defs = true`. Use `snake_case` for functions, variables, modules, and CLI flags. Use `PascalCase` for Pydantic and SQLAlchemy model classes. Keep MCP tool names action-oriented, such as `search_papers`, `get_page`, and `list_sections`.

## Testing Guidelines

Tests are expected under `tests/` and are discovered by Pytest. Name files `test_*.py` and test functions `test_*`. Add focused tests for parser, chunking, classification, and database changes. Keep environment validation in `test_setup.py`; do not require private Zotero data in unit tests.

## Commit & Pull Request Guidelines

Recent history uses concise Conventional Commit prefixes, especially `feat:` and `fix:`. Follow that style, for example `feat: add HTTP transport health check`. Pull requests should include a short summary, commands run, migration impact, and configuration changes such as new `.env` variables. Include logs when changing Docker, HTTP, or operational behavior.

## Security & Configuration Tips

Do not commit `.env`, bearer tokens, Zotero databases, PDFs, Chroma indexes, or generated logs. Keep defaults documented, but store local values in `.env`. When touching Docker deployment, preserve the explicit manual upload/import/restart workflow described in `docker/README.md`.
