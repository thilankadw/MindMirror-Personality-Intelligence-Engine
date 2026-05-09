"""API router definitions."""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes.auth import router as auth_router
from app.api.v1.routes.jobs import router as jobs_router
from app.api.v1.routes.predictions import router as predictions_router
from app.api.v1.routes.users import router as users_router


api_router = APIRouter()
api_router.include_router(auth_router, tags=["auth"])
api_router.include_router(jobs_router, tags=["inference"])
api_router.include_router(users_router, tags=["users"])
api_router.include_router(predictions_router, tags=["predictions"])
