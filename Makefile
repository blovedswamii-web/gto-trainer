# One-command local workflow. Requires: uv (or python3.12 + pip) and node/npm.
PY      := backend/.venv/bin/python
UV      := $(shell command -v uv 2>/dev/null)

.PHONY: install dev backend frontend test test-fast lint typecheck precompute clean

install: backend/.venv/.installed frontend/node_modules/.installed

backend/.venv/.installed: backend/pyproject.toml
ifdef UV
	cd backend && uv venv --allow-existing --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
else
	cd backend && python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev]"
endif
	touch $@

frontend/node_modules/.installed: frontend/package.json
	cd frontend && npm install
	touch $@

## dev: run API (:8000) and Vite (:5173) together; Ctrl-C stops both
dev: install
	@trap 'kill 0' INT TERM; \
	(cd backend && .venv/bin/uvicorn gto_trainer.api.app:app --reload --port 8000) & \
	(cd frontend && npm run dev) & \
	wait

backend: install
	cd backend && .venv/bin/uvicorn gto_trainer.api.app:app --reload --port 8000

frontend: install
	cd frontend && npm run dev

test: install
	cd backend && .venv/bin/python -m pytest -q

test-fast: install
	cd backend && .venv/bin/python -m pytest -q -m "not slow"

lint: install
	cd backend && .venv/bin/ruff check gto_trainer tests && .venv/bin/ruff format --check gto_trainer tests
	cd backend && .venv/bin/mypy gto_trainer tests
	cd frontend && npm run typecheck

typecheck: lint

## precompute: preflop equity table + CFR solutions for the postflop trainer (minutes)
precompute: install
	cd backend && .venv/bin/python -m gto_trainer.solver.precompute

clean:
	rm -rf backend/.venv frontend/node_modules frontend/dist backend/.pytest_cache backend/.mypy_cache
