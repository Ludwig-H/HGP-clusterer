import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .geometry import bary_weight_batch, union_if_adjacent_int

NB_THREADS_CGAL = 1 # Au delà de 8, les performances s'aplatissent

def unique_sorted_rows(arr: np.ndarray, *, sort_rows: bool = False) -> np.ndarray:
    a = np.asarray(arr, dtype=np.int64)
    if a.ndim != 2:
        raise ValueError("unique_sorted_rows expects a 2D array")
    if a.shape[0] == 0:
        return a.reshape(0, a.shape[1])
    if sort_rows:
        order = np.lexsort(a.T[::-1])
        a = np.ascontiguousarray(a[order])
    else:
        a = np.ascontiguousarray(a)
    keep = np.ones(a.shape[0], dtype=bool)
    if a.shape[0] > 1:
        keep[1:] = np.any(a[1:] != a[:-1], axis=1)
    return a[keep]


def _resolve_cgal_binary(dimension: int, weighted: bool, root: Path) -> Path:
    base = "EdgesCGALWeightedDelaunay" if weighted else "EdgesCGALDelaunay"
    if dimension in (2, 3):
        suffix = f"{dimension}D"
    else:
        suffix = "ND"
    binary = root / f"{base}{suffix}" / "build" / f"{base}{suffix}"
    if not binary.exists():
        raise FileNotFoundError(f"CGAL binary not found: {binary}. Run `scripts/setup_cgal.py`.")
    return binary


def edges_from_weighted_delaunay(points: np.ndarray, weights: np.ndarray | None = None, *, precision: str = "safe", root: Path | None = None) -> list[tuple[int, int]]:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2:
        raise ValueError("points must be a 2D array")
    weights_arr = None if weights is None else np.asarray(weights, dtype=np.float64)
    if weights_arr is not None and weights_arr.shape[0] != points.shape[0]:
        raise ValueError("weights must have the same length as points")
    dimension = points.shape[1]
    n_points = points.shape[0]
    root_dir = root or os.environ.get("CGALDELAUNAY_ROOT")
    if root_dir is None:
        root_dir = Path(__file__).resolve().parents[2] / "CGALDelaunay"
    else:
        root_dir = Path(root_dir)
    binary = _resolve_cgal_binary(dimension, weights_arr is not None, root_dir)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        points_file = tmp_path / f"points{n_points}.npy"
        output_file = tmp_path / f"edges{n_points}.npy"
        np.save(points_file, points)
        cmd: list[str] = [str(binary), str(points_file), str(output_file)]
        if weights_arr is not None:
            weights_file = tmp_path / "weights.npy"
            np.save(weights_file, weights_arr)
            cmd.insert(2, str(weights_file))
        env = os.environ.copy()
        if precision == "exact":
            env["CGAL_EXACT_PREDICATES"] = "1"
        env["CGAL_NTHREADS"] = str(NB_THREADS_CGAL)   # threads TBB pour CGAL
        # par prudence, évite la double-parallélisation ailleurs
        env["OMP_NUM_THREADS"] = "1"
        env["MKL_NUM_THREADS"] = "1"
        env["OPENBLAS_NUM_THREADS"] = "1"
        subprocess.run(cmd, check=True, env=env)
        edges = np.load(output_file)
    edges = np.asarray(edges, dtype=np.int64)
    if edges.size == 0:
        return []
    edges.sort(axis=1)
    edges = unique_sorted_rows(edges, sort_rows=False)
    return [(int(i), int(j)) for i, j in edges]


def _build_all_keys(combos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    combos = np.asarray(combos, dtype=np.int64)
    if combos.ndim != 2:
        raise ValueError("combos must be a 2D array")
    m, k = combos.shape
    if k < 2:
        return np.empty((0, 0), dtype=np.int64), np.empty((0,), dtype=np.int64)
    keys = []
    parents = []
    for r in range(k):
        mask = [c for c in range(k) if c != r]
        keys.append(combos[:, mask])
        parents.append(np.arange(m, dtype=np.int64))
    all_keys = np.vstack(keys).astype(np.int64, copy=False)
    parent_idx = np.concatenate(parents).astype(np.int64, copy=False)
    return all_keys, parent_idx


def orderk_delaunay3(
    M: np.ndarray,
    K: int,
    *,
    precision: str = "safe",
    verbose: bool = False,
    root: Path | None = None,
) -> list[list[int]]:
    M = np.ascontiguousarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("M must be 2D")
    if K < 1:
        raise ValueError("K must be >= 1")
    n, d = M.shape
    if n < 2:
        return []
    s2_all = (M * M).sum(axis=1)
    prev = edges_from_weighted_delaunay(M, precision=precision, root=root)
    if verbose:
        print("orderk_delaunay k = 1")
    if K == 1:
        return [list(edge) for edge in prev]
    prev_array = np.asarray(prev, dtype=np.int64)
    for k in range(2, K + 1):
        if prev_array.shape[0] < 2:
            return []
        combos = np.ascontiguousarray(prev_array, dtype=np.int64)
        Q = np.empty((combos.shape[0], d), dtype=np.float64)
        w = np.empty((combos.shape[0],), dtype=np.float64)
        bary_weight_batch(M, s2_all, combos, Q, w)
        if verbose:
            print("Computed weighted barycentres", Q.shape[0])
        edges = edges_from_weighted_delaunay(Q, w, precision=precision, root=root)
        if not edges:
            return []
        next_candidates: list[np.ndarray] = []
        buffer = np.empty(k + 1, dtype=np.int64)
        for i, j in edges:
            A = combos[i]
            B = combos[j]
            if union_if_adjacent_int(A, B, buffer):
                next_candidates.append(buffer.copy())
        if not next_candidates:
            return []
        prev_array = unique_sorted_rows(np.asarray(next_candidates, dtype=np.int64), sort_rows=False)
        if verbose:
            print(f"orderk_delaunay k = {k}")
    return prev_array.tolist()
