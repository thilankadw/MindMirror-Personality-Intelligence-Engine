"""Command-line entry points."""
# app/__main__.py
from __future__ import annotations

import importlib
import os

import uvicorn


# def main() -> None:
#     host = os.getenv("HOST", "0.0.0.0")
#     port = int(os.getenv("PORT", "8000"))
#     reload_enabled = os.getenv("RELOAD", "false").lower() in {"1", "true", "yes", "on"}

#     uvicorn.run("app.main:app", host=host, port=port, reload=reload_enabled)


# if __name__ == "__main__":
#     main()

def main() -> None:
    """Run the module entry point."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
    reload_enabled = os.getenv("RELOAD", "false").lower() == "true"
    app_target = "api.app.main:app"

    # When executed from `api/` directly, the top-level package is `app`.
    try:
        importlib.import_module("api.app.main")
    except ModuleNotFoundError:
        app_target = "app.main:app"

    uvicorn.run(
        app_target,
        host=host,
        port=port,
        reload=reload_enabled,
    )


if __name__ == "__main__":
    main()
