FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3.10-dev \
    git build-essential libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt /tmp/
RUN python3.10 -m pip install --upgrade pip && \
    python3.10 -m pip install -r /tmp/requirements.txt

COPY . /workspace
ENV PYTHONPATH=/workspace

CMD ["bash"]
