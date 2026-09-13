.PHONY: test test-official test-extra lint cpu-suite gpu-suite tables submission

test:            ## all CPU-runnable tests (official + extra)
	uv run pytest tests -q -p no:cacheprovider

test-official:   ## only the staff test suite
	uv run pytest tests/test_attention.py tests/test_ddp.py tests/test_sharded_optimizer.py tests/test_fsdp.py -q -p no:cacheprovider

test-extra:
	uv run pytest tests/test_extra_flash.py tests/test_extra_distributed.py tests/test_extra_model.py -q -p no:cacheprovider

test-triton-cpu: ## Triton kernels through the interpreter (Linux only)
	TRITON_INTERPRET=1 uv run pytest tests/test_extra_flash.py -k triton -q -p no:cacheprovider

lint:
	uvx ruff@0.15.10 check cs336_systems tests scripts

cpu-suite:
	scripts/run_cpu_suite.sh

gpu-suite:
	scripts/run_gpu_suite.sh

tables:          ## render results/*.jsonl|csv into writeup/tables.md
	uv run python scripts/make_tables.py

submission:
	./test_and_make_submission.sh
