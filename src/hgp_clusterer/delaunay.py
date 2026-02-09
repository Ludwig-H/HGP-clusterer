from __future__ import annotations

import numpy as np
from pathlib import Path
import warnings

# Try to import the binary extension
_import_error = None
try:
    from hgp_clusterer.geometry_binding import compute_delaunay
except ImportError as e:
    # Fallback or error message if not compiled
    compute_delaunay = None
    _import_error = e

def orderk_delaunay3(
    M: np.ndarray,
    K: int,
    *,
    precision: str = "safe",
    verbose: bool = False,
    backend: str = "geogram",
    root: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Order-K Delaunay triangulation using the optimized C++ binding.
    Returns (simplices, squared_radii).
    """
    if compute_delaunay is None:
        raise ImportError(
            f"The 'geometry_binding' extension is not loaded. Original error: {_import_error}. "
            "Please ensure the package is installed correctly with compiled extensions."
        )

    M = np.ascontiguousarray(M, dtype=np.float64) # Ensure float64 for C++ double* compatibility
    if M.ndim != 2:
        raise ValueError("M must be 2D")
    if K < 1:
        raise ValueError("K must be >= 1")
    n, d = M.shape
    if n < 2:
        return np.empty((0, K + 1), dtype=np.int32), np.empty(0, dtype=np.float64)

    # Call the C++ function directly
    # returns tuple(simplices, weights)
    try:
        # Note: We update C++ signature to accept backend string
        result = compute_delaunay(M, K, precision, verbose, backend)
    except Exception as e:
        raise RuntimeError(f"C++ Execution failed: {e}")

    # Check if result is a tuple (new version) or array (old version compatibility?)
    # We assume new version since we modified the C++ code.
    if isinstance(result, tuple):
        simplices, weights = result
    else:
        # Should not happen if C++ is updated
        simplices = result
        weights = np.zeros(simplices.shape[0], dtype=np.float64) 

    if simplices.size == 0:
        return np.empty((0, K + 1), dtype=np.int32), np.empty(0, dtype=np.float64)
        
    return simplices, weights