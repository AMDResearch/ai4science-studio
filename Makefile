# Validate model manifests, generated index, example scripts, and preflight --dry-run.
# Uses uv when available; otherwise a local .venv (gitignored).

export PYTHONPATH := tools
VENV ?= .venv
REQ := tools/ai4s_validate/requirements.txt
REQ_AI4S := tools/ai4s/requirements.txt

.PHONY: check check-fast fix test test-ai4s ensure-py ensure-ai4s

ifeq ($(shell command -v uv 2>/dev/null),)
PY := $(VENV)/bin/python
RUN = $(PY)
ensure-py:
	@if [ ! -x $(PY) ]; then python3 -m venv $(VENV) && $(VENV)/bin/pip install -q -r $(REQ); fi
ensure-ai4s: ensure-py
	@$(PY) -m pip install -q -r $(REQ_AI4S)
else
PY := uv
RUN = uv run --with-requirements $(REQ) python
RUN_AI4S = uv run --with-requirements $(REQ) --with-requirements $(REQ_AI4S) python
ensure-py:
	@true
ensure-ai4s:
	@true
endif

check: ensure-py ensure-ai4s
	$(RUN) -m ai4s_validate check
ifeq ($(shell command -v uv 2>/dev/null),)
	$(PY) -m ai4s validate
else
	$(RUN_AI4S) -m ai4s validate
endif

check-fast: ensure-py
	$(RUN) -m ai4s_validate check-fast

fix: ensure-py
	$(RUN) -m ai4s_validate fix

test: ensure-py ensure-ai4s
	$(RUN) -m unittest discover -s tools/ai4s_validate/tests -v
ifeq ($(shell command -v uv 2>/dev/null),)
	$(PY) -m unittest discover -s tools/ai4s/tests -v
else
	$(RUN_AI4S) -m unittest discover -s tools/ai4s/tests -v
endif

test-ai4s: ensure-ai4s
ifeq ($(shell command -v uv 2>/dev/null),)
	$(PY) -m unittest discover -s tools/ai4s/tests -v
else
	$(RUN_AI4S) -m unittest discover -s tools/ai4s/tests -v
endif
