.PHONY: install dev test lint typecheck run docker fmt

install:        ## Install core + OCR runtime
	pip install -e ".[ocr,dev]"

dev:            ## Install core + dev tools only (fast; no heavy models)
	pip install -e ".[dev]"

test:
	pytest -q

lint:
	ruff check src tests

fmt:
	ruff check --fix src tests

typecheck:
	mypy

run:            ## Run the API locally
	uvicorn ocr_service.app:app --reload --port 8000

docker:
	docker compose up --build
