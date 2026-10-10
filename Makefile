.PHONY: test lint gate
test:
	python3 -m pytest -m "not gpu" tests
lint:
	python3 -m ruff check --select E9,F63,F7,F82 buc ops
gate:
	python3 ops/harness/gate.py $(ID)
