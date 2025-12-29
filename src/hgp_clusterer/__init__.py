"""HypergraphPercol clustering package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .core import HypergraphPercol
from .estimator import HGPClusterer

__all__ = ["HypergraphPercol", "HGPClusterer"]


def __getattr__(name: str) -> Any:  # pragma: no cover - simple lazy import shim
    if name == "HypergraphPercol":
        module = import_module("hgp_clusterer.core")
        return module.HypergraphPercol
    raise AttributeError(f"module 'hgp_clusterer' has no attribute {name!r}")


def __dir__() -> list[str]:  # pragma: no cover - cosmetic helper
    return sorted(__all__)
