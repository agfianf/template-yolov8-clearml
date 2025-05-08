IMAGE_NAME=yolov8-custom:gpu-py3.10.11

run:
	docker run \
	--ipc=host \
	-it --rm --gpus all \
	-v ${PWD}:/workspace \
	-u $(id -u):$(id -g) \
	-e PYTHONPATH=/workspace \
	-w /workspace \
	-v ${PWD}/sawit.clearml.conf:/root/clearml.conf \
	$(IMAGE_NAME) \
	bash

build:
	docker build -t $(IMAGE_NAME) .

test_code:
	pytest tests -v 
