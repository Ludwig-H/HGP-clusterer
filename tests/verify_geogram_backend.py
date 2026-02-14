import numpy as np
import pytest
from hgp_clusterer.delaunay import orderk_delaunay3

def generate_points(n_points, dim, seed=42):
    rng = np.random.default_rng(seed)
    return rng.random((n_points, dim))

@pytest.mark.parametrize("dim", [2, 3, 4, 5])
@pytest.mark.parametrize("k", [1, 2, 3])
def test_geogram_backend_execution(dim, k):
    """
    Verifies that the Geogram backend executes correctly for various dimensions and K values.
    K=1 tests Standard Delaunay.
    K>1 tests Weighted Delaunay (Lifting).
    """
    n_points = 50 if dim < 4 else 20  # Keep it fast but non-trivial
    points = generate_points(n_points, dim)
    
    print(f"\n[Test] Dim={dim}, K={k}, N={n_points}")
    
    try:
        simplices, squared_radii = orderk_delaunay3(
            points, 
            K=k, 
            precision="safe", 
            verbose=True, 
            backend="geogram"
        )
        
        # Basic Validation
        assert simplices.ndim == 2
        assert simplices.shape[1] == k + 1
        assert squared_radii.ndim == 1
        assert len(simplices) == len(squared_radii)
        
        # Ensure we got some output (unless input is degenerate/empty, which random is not)
        if n_points >= k + 1:
            assert len(simplices) > 0, "Geogram backend returned empty result for valid input"
            
        print(f"[Success] Generated {len(simplices)} simplices.")
        
    except ImportError as e:
        pytest.skip(f"Geogram backend not available: {e}")
    except Exception as e:
        raise RuntimeError(f"Geogram backend crashed: {e}")

if __name__ == "__main__":
    # Manual run for quick debugging without pytest
    dims = [2, 3, 4, 5]
    ks = [2]
    
    for d in dims:
        for k in ks:
            try:
                test_geogram_backend_execution(d, k)
            except Exception as e:
                print(f"FAILED: {e}")
                exit(1)
    print("\nAll manual checks passed!")