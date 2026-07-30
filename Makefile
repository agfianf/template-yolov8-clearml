# The image reference is defined once, in src/params.py, and read back here -- so
# `make build` and the set_base_docker() call that every task inherits can never name
# different images. src/params.py takes the tag from ./VERSION. Override for a registry:
#   make build push TRAINER_IMAGE=ghcr.io/acme/yolo-trainer:0.2.2
# src/params.py imports nothing outside the stdlib, so this needs no venv.
IMAGE_NAME := $(shell PYTHONPATH=. python3 -c "from src.params import DOCKER_IMAGE; print(DOCKER_IMAGE)")
IMAGE_VERSION := $(lastword $(subst :, ,$(IMAGE_NAME)))

# Semantic version bump. Bump before any build whose image contents changed: a
# published tag must never be rebuilt, because every task that ran on it keeps
# requesting that exact tag.
#   make bump             -> 0.2.2 -> 0.2.3
#   make bump PART=minor  -> 0.2.2 -> 0.3.0
PART ?= patch

# ClearML Agent variables
TASK_ID ?=
QUEUE ?= default

# Print the resolved image reference -- use this to confirm what an override produced
# before spending a build on it.
image-name:
	@echo $(IMAGE_NAME)

bump:
	@PART=$(PART) python3 -c "import os, pathlib, sys; \
	part = os.environ['PART']; \
	sys.exit('PART must be major, minor or patch (got %r)' % part) if part not in ('major', 'minor', 'patch') else None; \
	f = pathlib.Path('VERSION'); \
	old = f.read_text().strip(); \
	n = [int(x) for x in old.split('.')]; \
	sys.exit('VERSION must be MAJOR.MINOR.PATCH (got %r)' % old) if len(n) != 3 else None; \
	new = {'major': (n[0] + 1, 0, 0), 'minor': (n[0], n[1] + 1, 0), 'patch': (n[0], n[1], n[2] + 1)}[part]; \
	f.write_text('.'.join(map(str, new)) + chr(10)); \
	print('%s -> %s' % (old, '.'.join(map(str, new))))"

run:
	PYTHONPATH=. uv run src/train.py

run-docker:
	docker run \
	--ipc=host \
	-it --rm --gpus all \
	-v ~/clearml.conf:/root/clearml.conf \
	-v ${PWD}:/workspace \
	-w /workspace \
	-e PYTHONPATH=/workspace \
	$(IMAGE_NAME) \
	bash

build:
	DOCKER_BUILDKIT=1 docker build \
	--build-arg IMAGE_VERSION=$(IMAGE_VERSION) \
	-t $(IMAGE_NAME) .

# Only meaningful once TRAINER_IMAGE points at a registry path; a bare name has
# nowhere to go. Agents on other hosts otherwise need `docker save | ssh | docker load`.
push:
	docker push $(IMAGE_NAME)

test_code:
	PYTHONPATH=. uv run pytest tests -v

# Everything except the tests that invoke a real exporter -- a few seconds, no GPU.
test_fast:
	PYTHONPATH=. uv run pytest tests -m "not slow"


# Must mirror the Dockerfile's `uv sync --frozen --no-dev`: this file is fed to
# Task.add_requirements(), so it defines the env a non-docker agent rebuilds.
# Do NOT add --resolution lowest-direct; that exports floor versions, not locked ones.
get-req:
	uv export \
	--format requirements.txt \
	--no-dev \
	--no-emit-project \
	--no-hashes \
	-o requirements.txt

# =============================================================================
# ClearML Agent Targets
# =============================================================================

# Start ClearML Agent daemon in Docker mode (uses pre-built image)
# Usage: make agent-daemon
# Usage: make agent-daemon QUEUE=gpu-queue
agent-daemon:
	clearml-agent daemon --queue $(QUEUE) --docker $(IMAGE_NAME) --gpus all

# Start ClearML Agent daemon in uv mode (no Docker, uses local Python/uv)
# Usage: make agent-daemon-uv
# Usage: make agent-daemon-uv QUEUE=cpu-queue
agent-daemon-uv:
	clearml-agent daemon --queue $(QUEUE)

# Execute a specific task locally (for debugging)
# Usage: make agent-execute TASK_ID=abc123
agent-execute:
ifndef TASK_ID
	$(error TASK_ID is required. Usage: make agent-execute TASK_ID=your_task_id)
endif
	clearml-agent execute --id $(TASK_ID) --docker $(IMAGE_NAME)

# Execute a specific task locally without Docker (uv mode)
# Usage: make agent-execute-uv TASK_ID=abc123
agent-execute-uv:
ifndef TASK_ID
	$(error TASK_ID is required. Usage: make agent-execute-uv TASK_ID=your_task_id)
endif
	clearml-agent execute --id $(TASK_ID)

# List running ClearML agents
agent-list:
	clearml-agent list

# Show agent daemon help
agent-help:
	clearml-agent daemon --help
