"""Utilities for logging."""
# app/core/logging.py
import logging
import sys
import uuid
from contextvars import ContextVar

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")

class RequestIdFilter(logging.Filter):
    """Provide request ID filter behavior."""
    def filter(self, record: logging.LogRecord) -> bool:
        """Filter the requested data."""
        record.request_id = request_id_ctx.get() or ""
        return True
    
def new_request_id() -> str:
    """Generate a new request ID."""
    return str(uuid.uuid4())

def setup_logging(log_level: str = "INFO", json_logs: bool = False) -> None:
    """Set up logging."""
    root_logger = logging.getLogger()

    # remove existing handlers
    while root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])

    root_logger.setLevel(log_level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(RequestIdFilter())

    # choose log format
    if json_logs:
        formatter = logging.Formatter(
            '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s",'
            '"msg":"%(message)s","request_id":"%(request_id)s"}'
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - [%(request_id)s] - %(levelname)s - %(message)s"
        )

    handler.setFormatter(formatter)

    root_logger.addHandler(handler)

# def setup_logging(level: str = "INFO", json_logs: bool = True) -> None:
#     root = logging.getLogger()
#     root.handlers.clear()
#     root.setLevel(level.upper())

#     handler = logging.StreamHandler(sys.stdout)
#     handler.addFilter(RequestIdFilter())

#     if json_logs:
#         fmt = (
#             '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s",'
#             '"msg":"%(message)s","request_id":"%(request_id)s"}'
#         )
#     else:
#         fmt = "%(asctime)s %(levelname)s [%(name)s] [req=%(request_id)s] %(message)s"

#     handler.setFormatter(logging.Formatter(fmt))
#     root.addHandler(handler)

# def new_request_id() -> str:
#     return uuid.uuid4().hex
