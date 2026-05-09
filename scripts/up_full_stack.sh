#!/usr/bin/env bash
set -euo pipefail

compose_files=(-f docker-compose.yml)
for extra in docker-compose.kafka.yml docker-compose.airflow.yml; do
  if [[ -f "$extra" ]]; then
    compose_files+=(-f "$extra")
  fi
done

docker compose "${compose_files[@]}" up -d --build

echo "API:      http://localhost:8000"
echo "Airflow:  http://localhost:8080 (admin/admin)"
echo "MLflow:   http://localhost:5000"
echo "Kafka:    localhost:9092"
echo "Postgres: localhost:5432"
