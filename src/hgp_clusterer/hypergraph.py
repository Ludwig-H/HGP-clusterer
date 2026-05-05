from __future__ import annotations

import itertools
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.metrics import pairwise_distances

from .delaunay import orderk_delaunay3
from .geometry import kth_radius, minimum_enclosing_ball

def _build_graph_KSimplexes(
    M: np.ndarray,
    K: int,
    min_samples: int,
    metric: str,
    complex_chosen: str,
    expZ: float,
    precision: str = "safe",
    verbose: bool = False,
    backend: str = "geogram",
    cgal_root: str | os.PathLike[str] | None = "../../CGALDelaunay",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    is_sparse_metric = metric == "sparse"
    if is_sparse_metric:
        M = np.asarray(M, dtype=np.float32)
        if M.ndim != 2 or M.shape[1] != 3:
            raise ValueError("For metric='sparse', M must be a list/array of (i, j, distance) triplets.")
        if M.size:
            n_points = int(np.max(M[:, :2])) + 1
        else:
            n_points = 0
        d = 0
    else:
        M = np.ascontiguousarray(M, dtype=np.float32)
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
            # Returns tuple (N_simplices, K+1), (N_simplices,)
            simplex_indices_arr, radii_arr = orderk_delaunay3(
                M, 
                min_samples - 1, 
                precision=precision, 
                verbose=verbose, 
                backend=backend,
                root=root_path
            )
            
            if simplex_indices_arr.dtype != np.int32:
                 simplex_indices_arr = simplex_indices_arr.astype(np.int32)
        except (FileNotFoundError, ImportError, RuntimeError) as exc:
            print(f"CRITICAL ERROR: CGAL/Geogram binding failed. Exc: {type(exc).__name__}: {exc}")
            if verbose:
                print(f"CGAL/Geogram binding not available or failed ({exc}). Falling back to Rips filtration.")
            complex_chosen = "rips"
        else:
            n_simplices = simplex_indices_arr.shape[0]
            if verbose:
                print(f"Simplexes sans filtration : {n_simplices}")
            
            if n_simplices > 0:
                # If weights were computed by C++, use them.
                if radii_arr is None or radii_arr.size == 0:
                     # Should not happen if CGAL works as expected
                     if verbose: print("Warning: CGAL returned no radii. Weights set to 1.0.")
                     radii_arr = np.ones(n_simplices, dtype=np.float64)
                
                # Check for fallback (radii_arr < 0 means C++ returned -1.0)
                fallback_mask = radii_arr < 0
                if np.any(fallback_mask):
                    if verbose:
                        print(f"Calculating {np.sum(fallback_mask)} radii natively using Miniball C++...")
                    
                    from ._cython import compute_fallback_radii
                    
                    if M.dtype != np.float64:
                        M = M.astype(np.float64)
                    if not M.flags.c_contiguous:
                        M = np.ascontiguousarray(M)
                        
                    if radii_arr.dtype != np.float64:
                        radii_arr = radii_arr.astype(np.float64)
                    if not radii_arr.flags.c_contiguous:
                        radii_arr = np.ascontiguousarray(radii_arr)
                    
                    # Call cython function which drops into C++ and runs OpenMP over mask
                    compute_fallback_radii(M, simplex_indices_arr, fallback_mask.view(np.uint8), radii_arr)
                
                # Apply Exponent if needed (radii_arr is squared radius)
                # target is radius^expZ.
                # r_sq = radius^2
                # radius^expZ = (r_sq)^(expZ/2)
                if expZ != 2:
                    # Avoid negative values if any numerical noise (should be positive)
                    radii_arr = np.maximum(radii_arr, 0)
                    radii_arr = radii_arr ** (expZ / 2.0)
                
                simplex_weights_arr = radii_arr.astype(np.float32)
                     
            else:
                 simplex_indices_arr = np.empty((0, K + 1), dtype=np.int32)
                 simplex_weights_arr = np.empty(0, dtype=np.float32)

    if complex_chosen.lower() != "orderk_delaunay":
        # ... (Legacy Gudhi Path) ...
        # Note: The Gudhi path still builds lists `flat_indices` and `weights`.
        # We need to unify the variable names at the end.
        
        import gudhi


        if is_sparse_metric:
            expZ_local = expZ * 2
            r2 = np.zeros(n, dtype=np.float32)
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
            simplex_indices_arr = np.array(flat_indices, dtype=np.int32).reshape(n_simplexes_list, K + 1)
            simplex_weights_arr = np.array(weights, dtype=np.float32)
        else:
            simplex_indices_arr = np.empty((0, K + 1), dtype=np.int32)
            simplex_weights_arr = np.empty(0, dtype=np.float32)

    # Convert to typed numpy arrays (ensure C-contiguity if needed, though Cython handles it)
    # The arrays are now ready for build_dual_graph_cython


    try:
        from ._cython import build_dual_graph_cython
        
        # New optimized return:
        # faces_unique_arr, e_u_arr, e_v_arr, e_w_arr, s_faces_arr, n_unique
        faces_unique, e_u, e_v, e_w, S_faces, n_unique = build_dual_graph_cython(
            simplex_indices_arr, simplex_weights_arr, K
        )
        
        return faces_unique, e_u, e_v, e_w, S_faces, n_unique
        
    except ImportError:

        # Fallback if Cython compilation failed or function missing
        if verbose:
             print("Warning: Cython build_dual_graph_cython not found. Using slow Python loop.")
        
        # Python Fallback with Deduplication
        # Map: tuple(sorted(face)) -> index
        face_map = {}
        unique_faces = []
        S_faces_list = []
        
        edges_u = []
        edges_v = []
        edges_w = []
        
        n_simplexes_total = simplex_indices_arr.shape[0]
        
        # Pre-allocate simplex_face_ids for current simplex
        simplex_face_ids = [0] * (K + 1)
        
        for i in range(n_simplexes_total):
            simplex = list(simplex_indices_arr[i])
            weight = float(simplex_weights_arr[i])
            
            # 1/r accumulation
            inv_w = 1.0 / weight if weight > 1e-12 else 1e12
            
            # 1. Identify/Create Faces
            for drop in range(K + 1):
                face = tuple(sorted([simplex[t] for t in range(K + 1) if t != drop]))
                
                if face in face_map:
                    fid = face_map[face]
                    S_faces_list[fid] += inv_w
                else:
                    fid = len(unique_faces)
                    face_map[face] = fid
                    unique_faces.append(face)
                    S_faces_list.append(inv_w)
                
                simplex_face_ids[drop] = fid
            
            # 2. Create Edges
            for idx in range(K):
                edges_u.append(simplex_face_ids[idx])
                edges_v.append(simplex_face_ids[idx+1])
                edges_w.append(weight)
        
        faces_unique = np.array(unique_faces, dtype=np.int32)
        e_u = np.array(edges_u, dtype=np.int32)
        e_v = np.array(edges_v, dtype=np.int32)
        e_w = np.array(edges_w, dtype=np.float32)
        S_faces = np.array(S_faces_list, dtype=np.float32)
        n_unique = len(unique_faces)
        
        return faces_unique, e_u, e_v, e_w, S_faces, n_unique
