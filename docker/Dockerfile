ARG PYTHON_IMAGE=python:3.13
FROM ${PYTHON_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md* ./
COPY src ./src

RUN pip install --upgrade pip \
    && pip install -e ".[dev]"

WORKDIR /workspace

CMD ["codepilot", "--help"]
