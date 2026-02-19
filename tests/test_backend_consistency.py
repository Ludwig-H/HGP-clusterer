import numpy as np
import pytest

from hgp_clusterer.delaunay import orderk_delaunay3


def _normalize(simplices: np.ndarray, radii: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if simplices.size == 0:
        return simplices.reshape(0, simplices.shape[1] if simplices.ndim == 2 else 0), radii

    simplices = np.sort(np.asarray(simplices, dtype=np.int32), axis=1)
    radii = np.asarray(radii, dtype=np.float64)

    order = np.lexsort([simplices[:, i] for i in range(simplices.shape[1] - 1, -1, -1)])
    return simplices[order], radii[order]


def _run_backend(points: np.ndarray, k: int, backend: str):
    try:
        simplices, radii = orderk_delaunay3(points, k, backend=backend, precision="safe", verbose=False)
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Backend '{backend}' unavailable in this environment: {exc}")
    return _normalize(simplices, radii)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("k", [1, 2])
def test_cgal_geogram_same_simplices_and_radii(dim: int, k: int):
    rng = np.random.default_rng(12345 + 10 * dim + k)
    # Keep coordinates in general position to reduce degeneracies.
    points = rng.normal(size=(160, dim)).astype(np.float64)

    cgal_s, cgal_r = _run_backend(points, k, "cgal")
    geogram_s, geogram_r = _run_backend(points, k, "geogram")

    assert cgal_s.shape == geogram_s.shape
    assert np.array_equal(cgal_s, geogram_s)

    assert cgal_r.shape == geogram_r.shape
    assert np.allclose(cgal_r, geogram_r, rtol=1e-6, atol=1e-7)
