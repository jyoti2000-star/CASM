# Makefile for CASM Clean

.PHONY: help test examples clean check-deps

# Default target
help:
	@echo "CASM - Clean Assembly Language Compiler"
	@echo "Available targets:"
	@echo "  help      - Show this help"
	@echo "  test      - Run test suite"
	@echo "  examples  - Compile all examples"
	@echo "  clean     - Clean output files"
	@echo "  check-deps - Check build dependencies"

# Run tests
test:
	@echo "Running CASM test suite..."
	python3 tests/test_casm.py

# Check build dependencies
check-deps:
	@echo "Checking build dependencies..."
	@python3 -c "import sys; sys.path.insert(0, 'src'); from utils.build import build_tools; build_tools.print_install_instructions()"

# Compile all examples
examples: check-deps
	@echo "Compiling example programs..."
	@mkdir -p output
	python3 casm.py asm examples/hello.casm
	python3 casm.py asm examples/variables.casm
	python3 casm.py asm examples/loop.casm
	python3 casm.py asm examples/input.casm
	python3 casm.py asm examples/conditional.casm
	@echo "Examples compiled to output/ directory"

# Clean output files
clean:
	@echo "Cleaning output files..."
	rm -rf output/*
	rm -rf __pycache__
	rm -rf src/__pycache__
	rm -rf src/core/__pycache__
	rm -rf src/stdlib/__pycache__
	rm -rf src/utils/__pycache__
	find . -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete"

# Quick demo
demo: check-deps
	@echo "Running hello world demo..."
	python3 casm.py run examples/hello.casm