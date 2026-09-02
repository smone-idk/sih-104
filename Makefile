# VoiceShield — dev entrypoints. `make dev` brings the whole thing up.
SHELL := /bin/bash
PY311 ?= python3.11
VENV  := backend/.venv
PY    := $(VENV)/bin/python
PIP   := $(VENV)/bin/pip

.PHONY: help venv install install-torch-cuda install-torch-cpu fetch-models \
        demo-assets seed baseline inventory api frontend dev test clean grep-honesty

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

venv: ## create the Python 3.11 virtualenv
	$(PY311) -m venv $(VENV)
	$(PIP) install -U pip wheel

install: venv ## install backend deps (torch installed separately)
	$(PIP) install -r backend/requirements.txt

install-torch-cuda: ## install CUDA build of torch (RTX 4060)
	$(PIP) install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

install-torch-cpu: ## install CPU build of torch (fallback path)
	$(PIP) install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu

fetch-models: ## download + cache all model weights into ./models
	$(PY) scripts/fetch_models.py

demo-assets: ## build the demo audio corpus (genuine tier; add --all for synthetic+cloned)
	$(PY) scripts/build_demo_assets.py --tier genuine

seed: ## create DB, seed directory + approval + voice profile
	$(PY) scripts/seed.py

baseline: ## compute the prosody human-baseline from genuine clips
	$(PY) scripts/build_baseline.py

inventory: ## load models once and print the detector inventory (Phase 0 gate)
	cd backend && .venv/bin/python -m voiceshield.inventory

api: ## run the FastAPI backend
	cd backend && .venv/bin/uvicorn voiceshield.api.app:app --host 127.0.0.1 --port 8000 --reload

frontend: ## run the Vite dev server
	cd frontend && npm run dev

dev: ## run backend + frontend together
	@$(MAKE) -j2 api frontend

test: ## run backend unit tests
	cd backend && .venv/bin/pytest -q

grep-honesty: ## fail if a hardcoded score timeline exists anywhere (§1, §16)
	@! grep -RInE '(score|risk)\s*=\s*[0-9]{2,}|if\s*\(?\s*t\s*[<>]=?\s*[0-9]+\s*\)?\s*(score|risk)' \
	  backend/voiceshield scripts frontend/src 2>/dev/null || \
	  (echo "FOUND a suspicious hardcoded score — investigate above" && exit 1)
	@echo "no hardcoded score timeline found"

clean: ## remove venv, caches
	rm -rf $(VENV) backend/**/__pycache__ .pytest_cache
