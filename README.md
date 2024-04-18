# Depth markov

## How to install dependencies

1. `pip install --upgrade pip-tools`
2. `pip-compile --generate-hashes --output-file=requirements.txt requirements.in --allow-unsafe`
3. `pip-sync requirements.txt requirements-dev.txt`

.PHONY: setup-dev pip-tools

PIP-TOOLS := $(shell command -v pip-compile 2> /dev/null)
pip-tools:
ifndef PIP-TOOLS
$(error "pip-tools not installed. Install by running 'pip install --upgrade pip-tools'")
endif

requirements.txt: pip-tools requirements.in
pip-compile --generate-hashes --output-file=requirements.txt requirements.in --allow-unsafe

requirements-dev.txt: pip-tools requirements-dev.in
pip-compile --generate-hashes --output-file=requirements-dev.txt requirements-dev.in --allow-unsafe

setup-dev: pip-tools requirements.txt requirements-dev.txt
pip-sync requirements.txt requirements-dev.txt
@echo "Setup complete, happy programming!"
