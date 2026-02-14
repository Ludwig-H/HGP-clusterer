import numpy as np
import sys
import os
import time

# Ensure we can import hgp_clusterer
sys.path.append(os.path.join(os.getcwd(), "src"))

from hgp_clusterer.delaunay import orderk_delaunay3

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def normalize_results(simplices, weights):
    if len(simplices) == 0:
        return np.array([]), np.array([])
        
    # 1. Sort vertices within each simplex row
    simplices_sorted = np.sort(simplices, axis=1)
    
    # 2. Sort the list of simplices lexicographically
    cols_to_sort = [simplices_sorted[:, i] for i in range(simplices_sorted.shape[1]-1, -1, -1)]
    sort_indices = np.lexsort(cols_to_sort)
    
    simplices_final = simplices_sorted[sort_indices]
    weights_final = weights[sort_indices]
    
    return simplices_final, weights_final

def run_comparison(dim, k, n_points, seed=42, dist_type="random", precision="safe"):
    print(f"{Colors.BOLD}Testing Dim={dim}, K={k}, N={n_points}, Type={dist_type}, Prec={precision}{Colors.ENDC} ... ", end="", flush=True)
    
    rng = np.random.default_rng(seed)
    
    if dist_type == "random":
        points = rng.random((n_points, dim))
    elif dist_type == "grid":
        # Create a grid
        side = int(n_points**(1/dim))
        if side < 2: side = 2
        lin = np.linspace(0, 1, side)
        grid = np.meshgrid(*([lin]*dim))
        points = np.vstack(list(map(np.ravel, grid))).T
        # Add tiny noise to avoid perfect degeneracy which might crash or be undefined
        points += rng.normal(0, 1e-10, points.shape) 
    
    # CGAL
    start = time.time()
    try:
        res_cgal = orderk_delaunay3(points, K=k, precision=precision, verbose=False, backend="cgal")
        simplices_cgal, weights_cgal = normalize_results(res_cgal[0], res_cgal[1])
        t_cgal = time.time() - start
    except Exception as e:
        print(f"\n{Colors.FAIL}[CGAL CRASH]{Colors.ENDC} {e}")
        return False

    # Geogram
    start = time.time()
    try:
        res_geo = orderk_delaunay3(points, K=k, precision=precision, verbose=False, backend="geogram")
        simplices_geo, weights_geo = normalize_results(res_geo[0], res_geo[1])
        t_geo = time.time() - start
    except Exception as e:
        print(f"\n{Colors.FAIL}[GEO CRASH]{Colors.ENDC} {e}")
        return False

    # Comparison
    fail_reasons = []
    
    # 1. Counts
    if len(simplices_cgal) != len(simplices_geo):
        fail_reasons.append(f"Count mismatch: CGAL={len(simplices_cgal)}, Geo={len(simplices_geo)}")
    
    # 2. Content
    elif len(simplices_cgal) > 0:
        if not np.array_equal(simplices_cgal, simplices_geo):
            diff = np.sum(np.any(simplices_cgal != simplices_geo, axis=1))
            fail_reasons.append(f"Simplex content mismatch: {diff} rows differ")
    
    # 3. Weights (Radii)
    # Check if CGAL returns valid weights
    check_weights = True
    if len(weights_cgal) > 0 and weights_cgal[0] == -1.0:
        # CGAL dD > 3 returns -1.0
        # Check if we expect this
        if dim > 3:
            check_weights = False
        else:
            fail_reasons.append("CGAL returned -1.0 weights in low dimension!")
    
    if check_weights and len(weights_cgal) > 0 and not fail_reasons:
        diffs = np.abs(weights_cgal - weights_geo)
        max_diff = np.max(diffs)
        if max_diff > 1e-6:
            fail_reasons.append(f"Weight mismatch max_diff={max_diff:.2e}")

    if not fail_reasons:
        print(f"{Colors.OKGREEN}[PASS]{Colors.ENDC} (CGAL: {t_cgal:.3f}s, Geo: {t_geo:.3f}s)")
        return True
    else:
        print(f"{Colors.FAIL}[FAIL]{Colors.ENDC}")
        for r in fail_reasons:
            print(f"  - {r}")
        return False

def main():
    print("=== Extensive Backend Comparison (CGAL vs Geogram) ===")
    
    # Standard 2D
    run_comparison(2, 1, 100)
    run_comparison(2, 2, 100)
    run_comparison(2, 3, 100)
    run_comparison(2, 5, 50) # Higher K

    # Standard 3D
    run_comparison(3, 1, 100)
    run_comparison(3, 2, 50)
    
    # 4D
    print(">> Retesting 4D with precision='exact'")
    run_comparison(4, 1, 30, precision="exact")
    run_comparison(4, 2, 20, precision="exact")

    # Grid (near degenerate)
    run_comparison(2, 1, 25, dist_type="grid") # 5x5 grid
    
    print("\nDone.")

if __name__ == "__main__":
    main()
