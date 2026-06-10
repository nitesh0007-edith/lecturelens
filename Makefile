.PHONY: dev test lint ingest-demo eval fmt

dev:
	docker-compose up

test:
	pytest -q

lint:
	ruff check backend/ eval/ scripts/

fmt:
	ruff format backend/ eval/ scripts/

ingest-demo:
	PYTHONPATH=backend python scripts/ingest_uofg.py

eval:
	PYTHONPATH=backend python eval/run_eval.py

build:
	docker build -t lecturelens .
