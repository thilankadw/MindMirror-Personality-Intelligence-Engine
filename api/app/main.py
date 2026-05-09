"""Application entry points."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.api.router import api_router  # noqa: E402
from app.core.config import settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402
from app.db import models  # noqa: F401,E402
from app.db.base import Base  # noqa: E402
from app.db.session import engine  # noqa: E402
from app.utils.middleware import RequestIdMiddleware  # noqa: E402


os.environ.setdefault("TORCH_WEIGHTS_ONLY", "0")
try:  # pragma: no cover - optional runtime compatibility path.
    import torch.serialization
    import transformers
    from sklearn.preprocessing import LabelEncoder
    from transformers.models.deberta_v2.tokenization_deberta_v2 import DebertaV2Tokenizer

    torch.serialization.add_safe_globals(
        [
            LabelEncoder,
            DebertaV2Tokenizer,
            transformers.PreTrainedTokenizerFast,
        ]
    )
except Exception:
    pass


logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create the FastAPI application."""
    setup_logging(
        log_level=settings.LOG_LEVEL,
        json_logs=settings.JSON_LOGS,
    )

    app = FastAPI(
        title=settings.APP_NAME,
        openapi_url=f"{settings.API_V1_PREFIX}/openapi.json",
        docs_url=f"{settings.API_V1_PREFIX}/docs",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RequestIdMiddleware)

    @app.on_event("startup")
    async def configure_async_db() -> None:
        """Configure the async database on startup."""
        logger.info(
            "app.startup.database.configure.start create_tables_on_startup=%s engine_configured=%s",
            settings.CREATE_TABLES_ON_STARTUP,
            engine is not None,
        )
        if engine is None:
            logger.warning("app.startup.database.configure.skipped reason=database_not_configured")
            return
        if settings.CREATE_TABLES_ON_STARTUP:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
            logger.info("app.startup.database.create_tables.done")
        else:
            logger.info("app.startup.database.create_tables.skipped")

    @app.get("/health", tags=["System"])
    async def health_check() -> dict[str, str]:
        """Return a simple process health payload."""
        return {"status": "healthy", "service": "mindmirror-personality"}

    app.include_router(api_router, prefix=settings.API_V1_PREFIX)
    return app


app = create_app()
