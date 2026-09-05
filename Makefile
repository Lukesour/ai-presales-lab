PYTHON ?= python3
PYTHONPATH := src

.PHONY: test lint demo eval benchmark-llama dify-check

test:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) -m pytest -q

lint:
	ruff check src scripts tests

demo:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_demo.py --case-id case-001

eval:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/run_eval.py --output data/results/offline-evaluation.json

benchmark-llama:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/benchmark_llama.py --output data/results/llama-benchmark.json

dify-check:
	./scripts/check_dify.sh
