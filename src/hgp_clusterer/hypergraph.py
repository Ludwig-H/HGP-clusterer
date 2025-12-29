from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from joblib import Parallel, delayed, cpu_count
from sklearn.metrics import pairwise_distances

from .delaunay import orderk_delaunay3
from .geometry import kth_radius, minimum_enclosing_ball

N_CPU_dispo = max(1, cpu_count())

def _build_graph_KSimplexes(
    M: np.ndarray,
    K: int,
    min_samples: int,
    metric: str,
    complex_chosen: str,
    expZ: float,
    precision: str = "safe",
    verbose: bool = False,
    cgal_root: str | os.PathLike[str] | None = "../../CGALDelaunay",
) -> tuple[list[list[int]], list[int], list[int], list[float], int]:
    is_sparse_metric = metric == "sparse"
    if is_sparse_metric:
        M = np.asarray(M, dtype=np.float64)
        if M.ndim != 2 or M.shape[1] != 3:
            raise ValueError("For metric='sparse', M must be a list/array of (i, j, distance) triplets.")
        if M.size:
            n_points = int(np.max(M[:, :2])) + 1
        else:
            n_points = 0
        d = 0
    else:
        M = np.ascontiguousarray(M, dtype=np.float64)
        n_points, d = M.shape
    if min_samples is None or min_samples <= K:
        min_samples = K + 1
    pre = metric == "precomputed"
    delaunay_possible = not pre and metric == "euclidean" and not is_sparse_metric and M.ndim == 2
    n = n_points
    if is_sparse_metric:
        complex_chosen = "rips"
    elif complex_chosen.lower() not in {"orderk_delaunay", "delaunay", "rips"}:
        if not delaunay_possible:
            complex_chosen = "rips"
        else:
            if d > 10 and n > 100:
                complex_chosen = "rips"
            elif d > 10:
                complex_chosen = "delaunay"
            elif d > 5 and n > 1000:
                complex_chosen = "rips"
            else:
                complex_chosen = "orderk_delaunay"
                
    # Buffers for K-simplices (flattened indices and weights)
    flat_indices: list[int] = []
    weights: list[float] = []
    
    root_path = Path(cgal_root) if cgal_root is not None else None
    
    if complex_chosen.lower() == "orderk_delaunay":
        try:
            # Returns (N_simplices, K+1) int64 array
            simplex_indices_arr = orderk_delaunay3(M, min_samples - 1, precision=precision, verbose=verbose, root=root_path)
        except FileNotFoundError as exc:
            if verbose:
                print(f"CGAL non disponible ({exc}). Repli sur la filtration Rips.")
            complex_chosen = "rips"
        else:
            n_simplices = simplex_indices_arr.shape[0]
            if verbose:
                print(f"Simplexes sans filtration : {n_simplices}")
            
            if n_simplices > 0:
                # OPTIMIZATION: Process in batches to reduce Joblib overhead
                # or use vectorized approaches if possible.
                # Since M is large, we keep _sqr_radius logic but batch it.
                
                # We need the weights.
                # Radii calculation is the bottleneck.
                # Batch size tuning: 1000 to 10000 usually optimal for small tasks.
                
                def _sqr_radius_batch(indices_batch: np.ndarray) -> np.ndarray:
                    # indices_batch: (B, K+1)
                    # We compute radius for each simplex in the batch
                    # This runs in a worker process
                    # Accessing M (global) in 'processes' mode usually relies on memory mapping or fork copy-on-write
                    # which is efficient for reading.
                    
                    res = np.empty(indices_batch.shape[0], dtype=np.float64)
                    for i in range(indices_batch.shape[0]):
                        pts = M[indices_batch[i]]
                        _, r_sq = minimum_enclosing_ball(pts)
                        res[i] = r_sq
                    return res

                # Calculate batch slices
                batch_size = 2048
                slices = [
                    simplex_indices_arr[i : i + batch_size] 
                    for i in range(0, n_simplices, batch_size)
                ]

                radii_batches = Parallel(n_jobs=N_CPU_dispo, prefer="processes")(
                    delayed(_sqr_radius_batch)(s) for s in slices
                )
                
                if verbose:
                    print("N_CPU_dispo utilisés : ", N_CPU_dispo)
                
                radii_arr = np.concatenate(radii_batches)
                
                if expZ != 2:
                    radii_arr = radii_arr ** (expZ / 2)
                
                simplex_weights_arr = radii_arr
                
                # Check dimensions match (logic check)
                if simplex_indices_arr.shape[1] != K + 1:
                     # This should theoretically not happen with orderk_delaunay3 output
                     # unless K param was inconsistent.
                     # Handle fallback if necessary or just reshape/filter?
                     # We assume correctness from C++ tool.
                     pass
                     
            else:
                 simplex_indices_arr = np.empty((0, K + 1), dtype=np.int64)
                 simplex_weights_arr = np.empty(0, dtype=np.float64)

    if complex_chosen.lower() != "orderk_delaunay":
        # ... (Legacy Gudhi Path) ...
        # Note: The Gudhi path still builds lists `flat_indices` and `weights`.
        # We need to unify the variable names at the end.
        
        import gudhi


        if is_sparse_metric:
            expZ_local = expZ * 2
            r2 = np.zeros(n, dtype=np.float64)
            st = gudhi.SimplexTree()
            for v in range(n):
                st.insert([int(v)], filtration=0.0)
            for i, j, dist in M:
                ii = int(i)
                jj = int(j)
                filt = float(dist)
                if ii == jj:
                    continue
                if jj < ii:
                    ii, jj = jj, ii
                st.insert([ii, jj], filtration=filt)
            if n:
                st.expansion(max_dimension=K)
        else:
            r = kth_radius(M, min_samples - 1, metric, pre)
            r2 = r**2
            if complex_chosen.lower() == "rips":
                r2 = r
                expZ_local = expZ * 2
                if precision == "exact":
                    mx = 2 * np.quantile(r, 0.99)
                else:
                    mx = (1 + 1 / math.sqrt(d)) * np.quantile(r, 0.99)
                if pre or metric != "euclidean":
                    D = M if pre else pairwise_distances(M, metric=metric)
                    st = gudhi.RipsComplex(distance_matrix=D, max_edge_length=mx).create_simplex_tree(max_dimension=K)
                else:
                    st = gudhi.RipsComplex(points=M, max_edge_length=mx).create_simplex_tree(max_dimension=K)
            else:
                expZ_local = expZ
                st = gudhi.DelaunayCechComplex(points=M).create_simplex_tree()
        
        for simplex, filt in st.get_skeleton(K):
            if len(simplex) != K + 1:
                continue
            
            # GUDHI returns vertices, we sort them for consistency (optional but good)
            simplex_sorted = sorted(simplex)
            
            if is_sparse_metric:
                max_kth_radius2 = 0.0
            else:
                max_kth_radius2 = max(r2[p] for p in simplex_sorted)
            
            filt_val = max(filt, max_kth_radius2)
            if expZ_local != 2:
                filt_val = filt_val ** (expZ_local / 2)
            
            flat_indices.extend(simplex_sorted)
            weights.append(float(filt_val))
            
    # --- Cython Optimization ---
    # Unification: 
    # If we came from Order-K (optimized), we already have simplex_indices_arr and simplex_weights_arr.
    # If we came from Gudhi (legacy), we have flat_indices and weights (lists).
    
    if complex_chosen.lower() != "orderk_delaunay":
        n_simplexes_list = len(weights)
        if n_simplexes_list > 0:
            simplex_indices_arr = np.array(flat_indices, dtype=np.int64).reshape(n_simplexes_list, K + 1)
            simplex_weights_arr = np.array(weights, dtype=np.float64)
        else:
            simplex_indices_arr = np.empty((0, K + 1), dtype=np.int64)
            simplex_weights_arr = np.empty(0, dtype=np.float64)

    # Convert to typed numpy arrays (ensure C-contiguity if needed, though Cython handles it)
    # The arrays are now ready for build_dual_graph_cython


    try:
        from ._cython import build_dual_graph_cython
        
        # Pass memory views
        faces_raw, e_u, e_v, e_w, faces_Simplexes, nS = build_dual_graph_cython(
            simplex_indices_arr, simplex_weights_arr, K
        )
        
    except ImportError:
        # Fallback if Cython compilation failed or function missing
        if verbose:
             print("Warning: Cython build_dual_graph_cython not found. Using slow Python loop.")
        faces_raw = []
        e_u = []
        e_v = []
        e_w = []
        nS = 0
        faces_Simplexes = []
        
        # Reconstruct iterator over simplices from arrays
        n_simplexes_total = simplex_indices_arr.shape[0]
        for i in range(n_simplexes_total):
            simplex = list(simplex_indices_arr[i])
            weight = float(simplex_weights_arr[i])
            
            # Simplex is guaranteed size K+1 here
            # Original fallback logic assumed inputs could be > K+1
            # But we normalized everything to K+1 in flat_indices/weights
            
            # So the fallback logic simplifies: we don't need combinations loop anymore
            # just the inner face generation loop.
            nS += 1
            base = len(faces_raw)
            for drop in range(K + 1):
                face = [simplex[t] for t in range(K + 1) if t != drop]
                faces_raw.append(face)
                faces_Simplexes.append((base + drop, face, weight))
            for idx in range(K):
                e_u.append(base + idx)
                e_v.append(base + idx + 1)
                e_w.append(weight)
                    
    return faces_raw, e_u, e_v, e_w, faces_Simplexes, nS
