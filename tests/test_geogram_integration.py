
import numpy as np
import pytest
from hgp_clusterer import HGPClusterer

def test_backend_cgal_default():
    """Test that the default backend (CGAL) works or at least is selected."""
    X = np.random.rand(20, 2)
    # Just checking initialization and basic fit attempt
    # If CGAL is compiled, this should run.
    clusterer = HGPClusterer(min_cluster_size=5, backend='cgal', verbose=True)
    try:
        clusterer.fit(X)
        print("CGAL backend fit success")
    except Exception as e:
        pytest.fail(f"CGAL fit failed: {e}")

def test_backend_geogram_available():
    """Test that requesting Geogram backend works when compiled."""
    X = np.random.rand(20, 2)
    clusterer = HGPClusterer(min_cluster_size=5, backend='geogram', verbose=True)
    
    try:
        clusterer.fit(X)
        print("Geogram backend fit success")
    except RuntimeError as e:
        pytest.fail(f"Geogram backend raised RuntimeError unexpectedly: {e}")
    except Exception as e:
        pytest.fail(f"Geogram backend raised unexpected exception: {e}")

if __name__ == "__main__":
    # Manual run if pytest not invoked
    try:
        test_backend_cgal_default()
        print("Test CGAL Default: PASS")
    except Exception as e:
        print(f"Test CGAL Default: FAIL {e}")
        
    try:
        test_backend_geogram_missing()
        print("Test Geogram Missing: PASS")
    except Exception as e:
        print(f"Test Geogram Missing: FAIL {e}")
