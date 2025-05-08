FROM python:3.11-slim

# Install linux packages
ENV DEBIAN_FRONTEND noninteractive
RUN apt update
RUN TZ=Etc/UTC apt install -y tzdata
RUN apt install --no-install-recommends -y gcc git zip curl htop libgl1-mesa-glx libglib2.0-0 libpython3-dev gnupg
# RUN alias python=python3

# Create working directory
WORKDIR /workspace

# Install pip packages
RUN pip install torch torchvision
RUN pip install --upgrade pip wheel
COPY requirements.txt /deps/requirements.txt
RUN pip install -r /deps/requirements.txt


# Set environment variables
ENV OMP_NUM_THREADS=1

# Cleanup
ENV DEBIAN_FRONTEND teletype