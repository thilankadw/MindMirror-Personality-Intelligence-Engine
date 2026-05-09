"""Utilities for middleware."""
# app/utils/middleware.py
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from app.core.logging import request_id_ctx, new_request_id

class RequestIdMiddleware(BaseHTTPMiddleware):
    """Provide request ID middleware behavior."""
    async def dispatch(self, request: Request, call_next):
        """Handle dispatch."""
        rid = request.headers.get("x-request-id") or new_request_id()
        token = request_id_ctx.set(rid)
        try:
            response = await call_next(request)
            response.headers["x-request-id"] = rid
            return response
        finally:
            request_id_ctx.reset(token)
