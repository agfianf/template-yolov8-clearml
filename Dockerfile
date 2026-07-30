FROM python:3.14-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.0 /uv /uvx /bin/

# The tag carries no stack information on purpose, so record it here instead. These
# are what to check when a task runs on CPU or dies on an unsupported arch:
#   docker inspect <image> --format '{{json .Config.Labels}}'
ARG IMAGE_VERSION=dev
LABEL org.opencontainers.image.title="yolo-trainer" \
      org.opencontainers.image.description="YOLO training runtime for ClearML agents" \
      org.opencontainers.image.source="https://github.com/agfianf/template-yolov8-clearml" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      io.clearml.python="3.14" \
      io.clearml.torch="2.9.1+cu128" \
      io.clearml.cuda="12.8" \
      io.clearml.driver-min="525.60.13" \
      io.clearml.cuda-arch="sm_70,sm_75,sm_80,sm_86,sm_90,sm_100,sm_120"

# Install dependencies
# Note: libgl1-mesa-glx was replaced by libgl1 in Debian Trixie
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl ca-certificates gcc git zip htop \
    libgl1 libglib2.0-0 \
    libpython3-dev gnupg \
    g++ && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
ENV PATH="/workspace/.venv/bin:$PATH"
ENV PYTHONPATH="/workspace"

# Install the application dependencies.
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-cache

# Copy the application into the container.
COPY ./src /workspace/src

# Deliberately runs as root. clearml-agent's docker-mode bootstrap needs it: it writes
# /etc/apt/apt.conf.d/, chowns /root/.cache/pip, copies /root/.ssh and runs apt-get.
# Under `USER nonroot` every one of those fails and the task hangs before reaching
# src/train.py -- verified against task f206e189 on NEW-gpu-machine-server2.
# Dropping the nonroot user also removes the `chown -R` that was duplicating the
# whole ~8GB venv into a second layer.