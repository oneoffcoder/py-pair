UV ?= uv
PYTHON ?= 3.13
VENV ?= .venv
UV_ENV = UV_PROJECT_ENVIRONMENT=$(VENV)
PROFILE_FLAGS ?= --instrument --output .profiles/pypair.prof --memory-output .profiles/pypair.memory.txt
PYPI_CONFIG ?= $(CURDIR)/.pypirc
PYPI_REPOSITORY ?= pypi
TWINE = $(UV) tool run --from twine twine

.DEFAULT_GOAL := help

.PHONY: help venv format lint test coverage check profile build publish-check publish publish-testpypi docs compile clean clean-venv

help:
	@printf '%s\n' \
		'make venv       Create $(VENV) and install project dependencies' \
		'make format     Run ruff format' \
		'make lint       Run ruff check' \
		'make test       Run pytest' \
		'make coverage   Run pytest with coverage and write HTML report to coverage/html' \
		'make check      Run lint and tests' \
		'make profile    Run the built-in cProfile workload (override with PROFILE_FLAGS=...)' \
		'make build      Build wheel and sdist with uv' \
		'make publish-check Build artifacts and run twine validation' \
		'make publish    Upload dist/* to PyPI using ./.pypirc (override with PYPI_CONFIG=...)' \
		'make publish-testpypi Upload dist/* to TestPyPI using ./.pypirc' \
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

coverage: $(VENV)
	@mkdir -p coverage
	$(UV_ENV) $(UV) run pytest --cov=pypair --cov-report=term-missing --cov-report=html:coverage/html

check: lint test

profile: $(VENV)
	@mkdir -p .profiles
	$(UV_ENV) $(UV) run python -m pypair.profiling $(PROFILE_FLAGS)

build: $(VENV)
	$(UV_ENV) $(UV) build

publish-check: $(VENV)
	$(UV_ENV) $(UV) build --clear
	$(TWINE) check dist/*

publish: publish-check
	@test -f "$(PYPI_CONFIG)" || (echo "Missing $(PYPI_CONFIG). Put your .pypirc in the repository root or override PYPI_CONFIG=..." >&2; exit 1)
	$(TWINE) upload --config-file "$(PYPI_CONFIG)" --repository "$(PYPI_REPOSITORY)" dist/*

publish-testpypi: PYPI_REPOSITORY = testpypi
publish-testpypi: publish

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
	rm -rf .profiles/
	rm -f .coverage
	rm -f .noseids

clean-venv:
	rm -rf $(VENV)
