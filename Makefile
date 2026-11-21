.PHONY: install test lint eval demo serve docker

install:
	pip install -e ".[dev,api,anthropic]"

test:
	pytest

lint:
	ruff check .

eval:
	hprag eval data/eval/halcyon_qa.jsonl --corpus data/corpus --compare-baseline --log-level WARNING --out eval_report.json

demo:
	hprag demo --log-level WARNING

serve:
	hprag serve --corpus data/corpus --port 8000

docker:
	docker build -t hprag .
