from __future__ import annotations

import numpy as np
from pathlib import Path
import warnings

# Try to import the binary extension
try:
    from hgp_clusterer.cgal_binding import compute_delaunay
except ImportError:
    # Fallback or error message if not compiled
    compute_delaunay = None

def orderk_delaunay3(
    M: np.ndarray,
    K: int,
    *,
    precision: str = "safe",
    verbose: bool = False,
    root: Path | None = None,
) -> np.ndarray:
    """
    Compute Order-K Delaunay triangulation using the optimized C++ binding.
    """
    if compute_delaunay is None:
        raise ImportError(
            "The 'cgal_binding' extension is not loaded. "
            "Please ensure the package is installed correctly with compiled extensions."
        )

    M = np.ascontiguousarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("M must be 2D")
    if K < 1:
        raise ValueError("K must be >= 1")
    n, d = M.shape
    if n < 2:
        return np.empty((0, K + 1), dtype=np.int64)

    # Call the C++ function directly
    # returns (M, K+1) array of int64
    try:
        result = compute_delaunay(M, K, precision, verbose)
    except Exception as e:
        raise RuntimeError(f"C++ Execution failed: {e}")

    if result.size == 0:
        return np.empty((0, K + 1), dtype=np.int64)
        
    return result