"""Union-Find data structure with a Cython-accelerated implementation."""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - exercised via compiled extension
    from ._cython import UnionFind as _CythonUnionFind  # type: ignore
except ImportError:  # pragma: no cover - fallback used when extension unavailable
    print("Warning: no cython module UnionFind. Python fallback.")
    class _CythonUnionFind:  # type: ignore[too-many-ancestors]
        def __init__(self, size: int) -> None:
            print("Warning: no cython module UnionFind. Python fallback.")
            if size < 0:
                raise ValueError("size must be non-negative")
            self.parent = np.arange(size, dtype=np.int64)
            self._size = np.ones(size, dtype=np.int64)

        def find(self, x: int) -> int:
            parent = self.parent
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return int(x)

        def union(self, a: int, b: int) -> bool:
            ra = self.find(a)
            rb = self.find(b)
            if ra == rb:
                return False
            size = self._size
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent = self.parent
            parent[rb] = ra
            size[ra] += size[rb]
            return True

        def component_size(self, x: int) -> int:
            return int(self._size[self.find(x)])


UnionFind = _CythonUnionFind

__all__ = ["UnionFind"]
