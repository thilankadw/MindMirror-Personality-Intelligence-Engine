#!/usr/bin/env bash
set -euo pipefail

compose_files=(-f docker-compose.yml)
for extra in docker-compose.kafka.yml docker-compose.airflow.yml; do
  if [[ -f "$extra" ]]; then
    compose_files+=(-f "$extra")
  fi
done

docker compose "${compose_files[@]}" down
