
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

def test_backend_geogram_missing():
    """Test that requesting Geogram backend raises appropriate error if not compiled."""
    X = np.random.rand(20, 2)
    clusterer = HGPClusterer(min_cluster_size=5, backend='geogram', verbose=True)
    
    # We expect a RuntimeError because we know Geogram was NOT compiled in this env
    # If by miracle it was compiled, this test would need adjustment, but here we expect failure.
    with pytest.raises(RuntimeError) as excinfo:
        clusterer.fit(X)
    
    print(f"Caught expected error: {excinfo.value}")
    assert "Geogram backend not compiled" in str(excinfo.value) or "Execution failed" in str(excinfo.value)

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
