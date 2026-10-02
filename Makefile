.PHONY: install lint test test-unit test-integration bench demo

install:  ## editable install with dev tools (CPU wheels work fine)
	pip install -e ".[dev]"

lint:
	ruff check .

test:  ## everything (integration tests launch real torchrun workers on CPU)
	pytest -q

test-unit:
	pytest -q -m "not integration"

test-integration:
	pytest -q -m integration

bench:  ## checkpoint + training-overhead benchmark (GPU if available)
	python scripts/benchmark.py --output benchmark_results.json

demo:  ## 2-worker elastic run of the GPT-2 example
	flextrain launch -c examples/gpt2_training/config.yaml --nproc-per-node 2 examples/gpt2_training/train_gpt2.py
