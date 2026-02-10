import numpy as np
import sys
import os

# Ensure we can import hgp_clusterer
sys.path.append(os.path.join(os.getcwd(), "src"))

from hgp_clusterer.delaunay import orderk_delaunay3

def normalize_results(simplices, weights):
    """
    Sorts simplices and their corresponding weights to allow strict comparison.
    """
    if len(simplices) == 0:
        return np.array([]), np.array([])
        
    # 1. Sort vertices within each simplex row
    simplices_sorted = np.sort(simplices, axis=1)
    
    # 2. Sort the list of simplices lexicographically
    # We use lexsort. Note that lexsort takes columns in reverse order of priority.
    # To sort by col 0, then col 1, etc., we pass columns in reverse.
    cols_to_sort = [simplices_sorted[:, i] for i in range(simplices_sorted.shape[1]-1, -1, -1)]
    sort_indices = np.lexsort(cols_to_sort)
    
    simplices_final = simplices_sorted[sort_indices]
    weights_final = weights[sort_indices]
    
    return simplices_final, weights_final

def compare_runs(dim, k, n_points=50, seed=42):
    print(f"\n--- Testing Dim={dim}, K={k}, N={n_points} ---")
    rng = np.random.default_rng(seed)
    points = rng.random((n_points, dim))
    
    # Run CGAL
    print("Running CGAL...")
    try:
        res_cgal = orderk_delaunay3(points, K=k, precision="safe", verbose=False, backend="cgal")
        simplices_cgal, weights_cgal = normalize_results(res_cgal[0], res_cgal[1])
        print(f"CGAL found {len(simplices_cgal)} simplices.")
    except Exception as e:
        print(f"CGAL Failed: {e}")
        return False

    # Run Geogram
    print("Running Geogram...")
    try:
        res_geo = orderk_delaunay3(points, K=k, precision="safe", verbose=False, backend="geogram")
        simplices_geo, weights_geo = normalize_results(res_geo[0], res_geo[1])
        print(f"Geogram found {len(simplices_geo)} simplices.")
    except Exception as e:
        print(f"Geogram Failed: {e}")
        return False

    # Compare Counts
    if len(simplices_cgal) != len(simplices_geo):
        print(f"[FAIL] Simplex counts differ! CGAL={len(simplices_cgal)}, Geo={len(simplices_geo)}")
        # Check intersection
        # Convert to set of tuples for intersection check
        set_cgal = set(map(tuple, simplices_cgal.tolist()))
        set_geo = set(map(tuple, simplices_geo.tolist()))
        
        only_cgal = len(set_cgal - set_geo)
        only_geo = len(set_geo - set_cgal)
        print(f"      In CGAL only: {only_cgal}")
        print(f"      In Geo only:  {only_geo}")
        
        return False
    
    if len(simplices_cgal) == 0:
        print("[WARN] Both returned empty result (N too small?). Matching.")
        return True

    # Compare Simplices (Exact Match expected for Generic Position)
    diff_simplices = np.sum(simplices_cgal != simplices_geo)
    if diff_simplices > 0:
        print(f"[FAIL] Simplices content mismatch! ({diff_simplices} discrepancies)")
        # Show first mismatch
        for i in range(len(simplices_cgal)):
            if not np.array_equal(simplices_cgal[i], simplices_geo[i]):
                print(f"Mismatch at index {i}:")
                print(f"  CGAL: {simplices_cgal[i]}")
                print(f"  Geo : {simplices_geo[i]}")
                break
        return False
    else:
        print("[PASS] Simplices match exactly.")

    # Compare Weights (Radii Squared)
    # Allow some tolerance due to different Numerics (CGAL Exact/Interval vs Eigen LDLT)
    # The algorithms are conceptually calculating the same geometric quantity (MEB radius).
    
    # Check max difference
    diffs = np.abs(weights_cgal - weights_geo)
    max_diff = np.max(diffs)
    
    # Relative error might be more relevant for large radii, but here points are in [0,1], so radii are small.
    # Absolute tolerance.
    tol = 1e-9
    
    if max_diff > tol:
        print(f"[WARN] Weights differ. Max Diff: {max_diff:.2e}")
        
        # Check how many are "significantly" different
        fail_indices = np.where(diffs > 1e-5)[0] # Loose tolerance for failure
        if len(fail_indices) > 0:
            print(f"[FAIL] {len(fail_indices)} weights differ by > 1e-5!")
            idx = fail_indices[0]
            print(f"  Index {idx}: CGAL={weights_cgal[idx]:.6f}, Geo={weights_geo[idx]:.6f}")
            
            # Debug MEB for this simplex
            simplex = simplices_cgal[idx]
            pts = points[simplex]
            print(f"  Simplex Points: \n{pts}")
            
            return False
        else:
            print(f"[PASS] Weights match within loose tolerance (1e-5).")
    else:
        print(f"[PASS] Weights match closely (diff < {tol}).")

    return True

if __name__ == "__main__":
    passed = True
    
    # 2D Tests
    if not compare_runs(dim=2, k=1): passed = False
    if not compare_runs(dim=2, k=2): passed = False
    
    # 3D Tests (Where new optimizations are active)
    if not compare_runs(dim=3, k=1): passed = False
    if not compare_runs(dim=3, k=2): passed = False
    
    if passed:
        print("\n=== GLOBAL SUCCESS: All comparisons passed ===")
        exit(0)
    else:
        print("\n=== GLOBAL FAILURE: Some comparisons failed ===")
        exit(1)
