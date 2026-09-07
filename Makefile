PYTHON ?= python3
PYTHONPATH := src

.PHONY: test lint demo eval benchmark-llama summarize-benchmarks dify-check

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

summarize-benchmarks:
	PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/summarize_benchmarks.py data/results/q4-metal-c1.json data/results/q8-metal-c1.json --output data/results/q4-q8-summary.json

dify-check:
	./scripts/check_dify.sh
