"""HypergraphPercol clustering package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["HypergraphPercol"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy import shim
    if name == "HypergraphPercol":
        module = import_module("hypergraphpercol.core")
        return module.HypergraphPercol
    raise AttributeError(f"module 'hypergraphpercol' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper
    return sorted(__all__)
