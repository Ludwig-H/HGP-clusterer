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

N_CPU_dispo = max(1, cpu_count() - 1)

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
    Simplexes: list[tuple[list[int], float]] = []
    root_path = Path(cgal_root) if cgal_root is not None else None
    if complex_chosen.lower() == "orderk_delaunay":
        try:
            simplexes = orderk_delaunay3(M, min_samples - 1, precision=precision, verbose=verbose, root=root_path)
        except FileNotFoundError as exc:
            if verbose:
                print(f"CGAL non disponible ({exc}). Repli sur la filtration Rips.")
            complex_chosen = "rips"
        else:
            if verbose:
                print(f"Simplexes sans filtration : {len(simplexes)}")
            if simplexes:
                def _sqr_radius(simplex: Sequence[int]) -> float:
                    pts = M[np.asarray(simplex, dtype=np.int64)]
                    _, radius_sq = minimum_enclosing_ball(pts)
                    return radius_sq
                radii_sq = Parallel(n_jobs=N_CPU_dispo, prefer="processes")(
                    delayed(_sqr_radius)(s) for s in simplexes
                )
                if verbose:
                    print("N_CPU_dispo utilisés : ", N_CPU_dispo)
                if expZ != 2:
                    radii_sq = np.asarray(radii_sq, dtype=np.float64) ** (expZ / 2)
                Simplexes = [(list(s), float(radii_sq[i])) for i, s in enumerate(simplexes)]
    if complex_chosen.lower() != "orderk_delaunay":
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
            simplex = list(sorted(simplex))
            if is_sparse_metric:
                max_kth_radius2 = 0.0
            else:
                max_kth_radius2 = max(r2[p] for p in simplex)
            filt = max(filt, max_kth_radius2)
            if expZ_local != 2:
                filt = filt ** (expZ_local / 2)
            Simplexes.append((simplex, float(filt)))
    faces_raw: list[list[int]] = []
    e_u: list[int] = []
    e_v: list[int] = []
    e_w: list[float] = []
    nS = 0
    faces_Simplexes = []
    for simplex, weight in Simplexes:
        if len(simplex) <= K:
            continue
        for vertices in itertools.combinations(range(len(simplex)), K + 1):
            nS += 1
            vert = tuple(sorted(vertices))
            base = len(faces_raw)
            for drop in range(K + 1):
                face = [simplex[vert[i]] for i in range(K + 1) if i != drop]
                faces_raw.append(face)
                faces_Simplexes.append((base + drop, face, float(weight)))
            for idx in range(K):
                e_u.append(base + idx)
                e_v.append(base + idx + 1)
                e_w.append(float(weight))
    return faces_raw, e_u, e_v, e_w, faces_Simplexes, nS
