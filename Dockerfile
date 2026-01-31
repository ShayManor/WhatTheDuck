ARG BASE_IMAGE=ghcr.io/coreweave/ml-containers/torch-extras:17ad6db-base-cuda12.9.1-ubuntu22.04-torch2.10.0-vision0.25.0-audio2.10.0-abi1
FROM ${BASE_IMAGE} AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Minimal OS deps
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/* \

RUN python -m pip install --upgrade "pip>=24.0" && \
    pip install cudaq

COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

# Classiq SDK stores auth config at ~/.config/classiq/config.env by default; mount it at runtime.
RUN mkdir -p /root/.config/classiq

CMD ["python", "sweep.py", "--help"]
