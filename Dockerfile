FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH=/app/api:/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv \
    && pip install --upgrade pip setuptools wheel

COPY requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY api /app/api
COPY src /app/src
COPY utils /app/utils
COPY pipelines /app/pipelines
COPY config.yaml /app/config.yaml

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]