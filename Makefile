IMAGE_NAME=yolov8-custom:gpu-py3.10.11
IMAGE_NAME_v2=yolov11-binsho:py3.12

run:
	docker run \
	--ipc=host \
	-it --rm --gpus all \
	-v ${PWD}:/workspace \
	-u $(id -u):$(id -g) \
	-e PYTHONPATH=/workspace \
	-w /workspace \
	-v ${PWD}/ikan.clearml.conf:/root/clearml.conf \
	$(IMAGE_NAME) \
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
	-o requirements-task.txt