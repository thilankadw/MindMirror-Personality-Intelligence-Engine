#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DATABASE_URL:-}" ]]; then
  echo "DATABASE_URL is not set."
  exit 1
fi

if command -v alembic >/dev/null 2>&1; then
  alembic upgrade head
else
  python -m alembic upgrade head
fi

echo "Alembic migrations applied to DATABASE_URL target."
