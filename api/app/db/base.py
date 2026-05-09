"""Base classes and shared interfaces."""
# app/db/base.py
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Provide base behavior."""
    pass
