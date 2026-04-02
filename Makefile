UV ?= uv
PYTHON ?= 3.13
VENV ?= .venv
UV_ENV = UV_PROJECT_ENVIRONMENT=$(VENV)

.DEFAULT_GOAL := help

.PHONY: help venv format lint test check build docs compile clean clean-venv

help:
	@printf '%s\n' \
		'make venv       Create $(VENV) and install project dependencies' \
		'make format     Run ruff format' \
		'make lint       Run ruff check' \
		'make test       Run pytest' \
		'make check      Run lint and tests' \
		'make build      Build wheel and sdist with uv' \
		'make docs       Build Sphinx docs' \
		'make compile    Compile Python sources' \
		'make clean      Remove caches and build artifacts' \
		'make clean-venv Remove $(VENV)'

$(VENV):
	$(UV) venv --python $(PYTHON) $(VENV)
	$(UV_ENV) $(UV) sync

venv: $(VENV)

format: $(VENV)
	$(UV_ENV) $(UV) run ruff format .

lint: $(VENV)
	$(UV_ENV) $(UV) run ruff check .

test: $(VENV)
	$(UV_ENV) $(UV) run pytest

check: lint test

build: $(VENV)
	$(UV_ENV) $(UV) build

docs: $(VENV)
	$(UV_ENV) $(UV) sync --group docs
	SPHINXBUILD="$(CURDIR)/$(VENV)/bin/sphinx-build" $(MAKE) -C docs html

compile: $(VENV)
	$(UV_ENV) $(UV) run python -m compileall -f ./pypair

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	rm -rf coverage/
	rm -rf dist/
	rm -rf build/
	rm -rf pypair.egg-info/
	rm -rf pypair/pypair.egg-info/
	rm -rf jupyter/.ipynb_checkpoints/
	rm -rf joblib_memmap/
	rm -rf docs/build/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	rm -f .coverage
	rm -f .noseids

clean-venv:
	rm -rf $(VENV)
