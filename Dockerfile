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

# -m gives nonroot a real home. Without it, libraries that write per-user config
# (ultralytics, matplotlib) fail on /home/nonroot and fall back to /tmp -- matplotlib
# warns this slows imports and hurts multiprocessing, which affects dataloader workers.
RUN groupadd -r nonroot && useradd -r -m -g nonroot nonroot

# Install the application dependencies.
# chown runs inside this same layer on purpose: as a separate `RUN chown -R`, Docker's
# copy-on-write duplicates the whole ~8GB venv into a second layer, doubling the image.
RUN --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-cache \
    && chown -R nonroot:nonroot /workspace/.venv \
    && chown nonroot:nonroot /workspace

# Copy the application into the container.
COPY --chown=nonroot:nonroot ./src /workspace/src

USER nonroot