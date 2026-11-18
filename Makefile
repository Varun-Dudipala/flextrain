.PHONY: install lint test test-unit test-integration chaos soak bench demo

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

chaos:  ## randomized fault injection: SIGKILL random ranks, require a bitwise-identical result
	python -m tests.integration.chaos --trials 50 --max-kills 3 --output chaos_results.json

soak:  ## GPT-2 small on 2 ranks, killed at random for 700 steps; must end bitwise identical to an unkilled run (takes hours)
	python scripts/soak.py --steps 700 --ckpt-every 25 --mean-kill-seconds 480 --min-kill-gap 120 --output soak_results.json

bench:  ## checkpoint + training-overhead benchmark (GPU if available)
	python scripts/benchmark.py --output benchmark_results.json

demo:  ## 2-worker elastic run of the GPT-2 example
	flextrain launch -c examples/gpt2_training/config.yaml --nproc-per-node 2 examples/gpt2_training/train_gpt2.py
