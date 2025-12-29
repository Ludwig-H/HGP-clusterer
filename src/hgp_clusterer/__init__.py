"""HypergraphPercol clustering package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .core import HGPClusterer

__all__ = ["HGPClusterer"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy import shim
    raise AttributeError(f"module 'hgp_clusterer' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper
    return sorted(__all__)
