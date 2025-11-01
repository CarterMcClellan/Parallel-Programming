.PHONY: profile test clean autotune help

profile:
	@echo "Running profiler..."
	@POPCORN_FD=1 python eval.py profile test_cases.txt > profile_output.txt
	@echo "\n=== Profile Report ==="
	@grep "benchmark.0.report:" profile_output.txt | cut -d' ' -f2 | python -m base64 -d
	@echo "\n=== Full output saved to profile_output.txt ==="

test:
	@echo "Running tests..."
	@POPCORN_FD=1 python eval.py test test_cases.txt

autotune:
	@echo "Auto-tuning BLOCK_SIZE..."
	@python autotune.py

clean:
	@rm -f profile_output.txt
	@echo "Cleaned up profile output"

help:
	@echo "Available targets:"
	@echo "  make test     - Run correctness tests"
	@echo "  make profile  - Profile kernel performance"
	@echo "  make autotune - Test all BLOCK_SIZE values and show performance breakdown"
	@echo "  make clean    - Clean up output files"