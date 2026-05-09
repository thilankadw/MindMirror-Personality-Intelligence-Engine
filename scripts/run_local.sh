#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example"
fi

compose_files=(-f docker-compose.yml)
for extra in docker-compose.kafka.yml docker-compose.airflow.yml; do
  if [[ -f "$extra" ]]; then
    compose_files+=(-f "$extra")
  fi
done

docker compose "${compose_files[@]}" up --build -d

echo "Stack started. API: http://localhost:8000, Airflow: http://localhost:8080"
