# Validate model manifests, generated index, example scripts, and preflight --dry-run.
# Uses uv when available; otherwise a local .venv (gitignored).

export PYTHONPATH := tools
VENV ?= .venv
REQ := tools/ai4s_validate/requirements.txt

.PHONY: check check-fast fix test

ifeq ($(shell command -v uv 2>/dev/null),)
PY := $(VENV)/bin/python
RUN = $(PY)
ensure-py:
	@if [ ! -x $(PY) ]; then python3 -m venv $(VENV) && $(VENV)/bin/pip install -q -r $(REQ); fi
else
PY := uv
RUN = uv run --with-requirements $(REQ) python
ensure-py:
	@true
endif

check: ensure-py
	$(RUN) -m ai4s_validate check

check-fast: ensure-py
	$(RUN) -m ai4s_validate check-fast

fix: ensure-py
	$(RUN) -m ai4s_validate fix

test: ensure-py
	$(RUN) -m unittest discover -s tools/ai4s_validate/tests -v
