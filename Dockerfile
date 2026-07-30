FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.26 /uv /uvx /bin/

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