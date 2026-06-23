PROJECT_NAME = AIGear
PYTHON_VERSION = 3.10
PYTHON_INTERPRETER = python

## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Run reproducible evaluation
.PHONY: evaluate
evaluate:
	$(PYTHON_INTERPRETER) -m algear.evaluate $(ARGS)

## Run tests
.PHONY: test
test:
	python -m pytest tests

## Run inference API server
.PHONY: serve
serve:
	$(PYTHON_INTERPRETER) -m algear.api

## Build and run Docker container
.PHONY: docker-up
docker-up:
	docker compose up --build

## Stop Docker container
.PHONY: docker-down
docker-down:
	docker compose down

## Make dataset
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) algear/dataset.py

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
