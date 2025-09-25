# Repository Guidelines

## Project Structure & Module Organization
- `backend/` — FastAPI app (`app/`), config in `app/config/settings.py`, entrypoint `start_server.py`, deps in `requirements.txt`.
- `frontend/` — vanilla JS/CSS/HTML (`js/`, `css/`, `index.html`).
- `tests/` — backend pytest in `tests/backend/...`, frontend Jest tests in `tests/frontend/...` (reports under `tests/reports/`).
- `agents/` — reference agent design notes; non-runtime docs.

## Build, Test, and Development Commands
- Python setup: `python -m venv .venv` then activate (`.venv\Scripts\Activate.ps1` on Windows, `source .venv/bin/activate` on Unix) and `pip install -r backend/requirements.txt`.
- Environment: copy `.env` template (`copy backend\.env.example backend\.env` on Windows, or `cp backend/.env.example backend/.env`).
- Run backend: `python backend/start_server.py` (API at `/api/v1/*`, static `frontend/` served if present).
- All tests (recommended): `python tests/run_tests.py`
- Backend-only: `python -m pytest tests/backend -q --cov=backend/app`
- Frontend-only: ensure Node 18+, then `python tests/run_tests.py --frontend-only` (auto-generates `frontend/package.json` if missing).

## Coding Style & Naming Conventions
- Python: PEP 8, 4-space indent, type hints. Modules/functions `snake_case`, classes `PascalCase`. Format with `black` and `isort`.
  - Example: `black backend/app tests` and `isort backend/app tests`.
- JavaScript: 2-space indent. Components in `frontend/js/components/` use `PascalCase.js`; utilities in `frontend/js/utils/` use `lowercase.js`.

## Testing Guidelines
- Python: `pytest` with markers (`unit`, `integration`, `system`, etc.). Coverage threshold 70% (see `pytest.ini`). Place tests under `tests/backend/...` mirroring package paths.
- JS: `jest` with `jsdom`. Tests live in `tests/frontend/...`. Coverage written to `tests/reports/frontend-coverage/`.

## Commit & Pull Request Guidelines
- Commit messages: use Conventional Commits (e.g., `feat:`, `fix:`, `docs:`, `chore:`). Current history is sparse—adopt this going forward.
- PRs must include: concise description, linked issue, test plan/results (`python tests/run_tests.py`), screenshots for UI changes, and notes on any config/.env changes. CI runs on PRs.

## Security & Configuration
- Never commit secrets. Use `backend/.env.example` to create local `.env` and set LLM/RAG keys as needed. Validate with `app/config/settings.py` helpers.

