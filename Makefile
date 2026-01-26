IMAGE_NAME=yolov8-custom:gpu-py3.10.11
IMAGE_NAME_v2=yolov11-binsho:py3.12

# ClearML Agent variables
TASK_ID ?=
QUEUE ?= default

run:
	PYTHONPATH=. uv run src/train.py

run-docker:
	docker run \
	--ipc=host \
	-it --rm --gpus all \
	-v ${PWD}:/workspace \
	-u $(id -u):$(id -g) \
	-e PYTHONPATH=/workspace \
	-w /workspace \
	-v ${PWD}/clearml.conf:/root/clearml.conf \
	$(IMAGE_NAME) \
	bash

run-docker-v2:
	docker run \
	--ipc=host \
	-it --rm --gpus all \
	-v ~/clearml.conf:/root/clearml.conf \
	-v ${PWD}:/workspace \
	-w /workspace \
	-e PYTHONPATH=/workspace \
	$(IMAGE_NAME_v2) \
	bash

build:
	docker build -t $(IMAGE_NAME) .

build-v2:
	DOCKER_BUILDKIT=1  docker build \
	-f Dockerfile.v2 \
	-t $(IMAGE_NAME_v2) .

test_code:
	pytest tests -v 


get-req:
	uv export \
	--format requirements.txt \
	--resolution lowest-direct \
	--no-hashes \
	-o requirements.txt

# =============================================================================
# ClearML Agent Targets
# =============================================================================

# Start ClearML Agent daemon in Docker mode (uses pre-built image)
# Usage: make agent-daemon
# Usage: make agent-daemon QUEUE=gpu-queue
agent-daemon:
	clearml-agent daemon --queue $(QUEUE) --docker $(IMAGE_NAME_v2) --gpus all

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
	clearml-agent execute --id $(TASK_ID) --docker $(IMAGE_NAME_v2)

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
