from __future__ import annotations

import numpy as np
from collections import Counter

try:  # pragma: no cover - prefer compiled implementations
    from ._cython import (  # type: ignore
        bary_weight_batch as _cython_bary_weight_batch,
        bary_weight_one as _cython_bary_weight_one,
        union_if_adjacent_int as _cython_union_if_adjacent_int,
    )
except ImportError:  # pragma: no cover - fallback used when extension unavailable
    print("Warning: no cython modules for bary_weight_batch, bary_weight_one and. Python fallback.")
    _cython_bary_weight_one = None
    _cython_bary_weight_batch = None
    _cython_union_if_adjacent_int = None

# Optional KNN backends
try:
    import faiss
    _HAS_FAISS = True
except ImportError:
    _HAS_FAISS = False

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


def fast_knn_indices(X_train: np.ndarray, X_query: np.ndarray, k: int, metric: str = "euclidean") -> tuple[np.ndarray, np.ndarray]:
    """
    Find k-nearest neighbors using the fastest available backend (FAISS > SciPy > Sklearn).
    Returns (distances, indices).
    """
    X_train = np.ascontiguousarray(X_train, dtype=np.float32)
    X_query = np.ascontiguousarray(X_query, dtype=np.float32)
    n_samples, d = X_train.shape
    
    # 1. FAISS (Euclidean only, but extremely fast)
    if _HAS_FAISS and metric == "euclidean":
        # IndexFlatL2 is exact search. For very large N, IVFFlat could be used but requires training.
        # For clustering sub-samples, exact L2 is usually fine and safe.
        index = faiss.IndexFlatL2(d)
        # Check if GPU is available? For now, CPU FAISS is already huge win.
        index.add(X_train)
        D, I = index.search(X_query, k)
        # FAISS returns squared distances
        return np.sqrt(D), I

    # 2. SciPy cKDTree (Euclidean/Minkowski)
    if _HAS_SCIPY and metric == "euclidean":
        tree = cKDTree(X_train)
        D, I = tree.query(X_query, k=k, workers=-1)
        return D, I

    # 3. Sklearn (Fallback for other metrics or if deps missing)
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, metric=metric, n_jobs=-1)
    nn.fit(X_train)
    D, I = nn.kneighbors(X_query)
    return D, I


def propagate_labels_knn(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int = 5, metric: str = "euclidean") -> np.ndarray:
    """
    Propagate labels from X_train to X_query using k-NN majority vote.
    """
    if k <= 0:
        raise ValueError("k must be >= 1")
    
    # Special case: 1-NN (no voting needed, faster)
    if k == 1:
        _, indices = fast_knn_indices(X_train, X_query, k=1, metric=metric)
        # indices shape (n_query, 1) or (n_query,) depending on backend? 
        # FAISS returns (n, k), SciPy (n,) if k=1 but we asked for k.
        # Let's ensure shape.
        indices = indices.reshape(X_query.shape[0], -1)
        return y_train[indices[:, 0]]

    _, indices = fast_knn_indices(X_train, X_query, k=k, metric=metric)
    
    # Majority Vote
    # This can be slow in pure Python for millions of points.
    # We can use SciPy mode if available, or a simple bincount per row.
    # Since k is small (e.g. 5), a loop is acceptable or we try to vectorize.
    
    n_query = X_query.shape[0]
    y_pred = np.empty(n_query, dtype=y_train.dtype)
    
    # Get neighbors labels: shape (n_query, k)
    neighbor_labels = y_train[indices]
    
    # Fast voting for small k
    from scipy.stats import mode
    mode_result = mode(neighbor_labels, axis=1, keepdims=False)
    # SciPy < 1.11 returns ModeResult.mode as array. 
    # SciPy >= 1.11 uses keepdims.
    # Let's handle return type safely.
    if hasattr(mode_result, 'mode'):
        res = mode_result.mode
        if np.ndim(res) > 1:
            y_pred = res.flatten()
        else:
            y_pred = res
    else:
        # Fallback if scipy mode behaves unexpectedly
        for i in range(n_query):
            counts = Counter(neighbor_labels[i])
            y_pred[i] = counts.most_common(1)[0][0]
            
    return y_pred


def bary_weight_one(
    M: np.ndarray,
    s2_all: np.ndarray,
    idx: np.ndarray,
    out_q: np.ndarray,
) -> float:
    if _cython_bary_weight_one is not None:
        return float(_cython_bary_weight_one(M, s2_all, idx, out_q))
    print("Warning: no cython module bary_weight_one. Python fallback.")
    idx_arr = np.asarray(idx, dtype=np.int64)
    points = M[idx_arr]
    np.copyto(out_q, points.mean(axis=0))
    qnorm2 = float(np.dot(out_q, out_q))
    return qnorm2 - float(np.asarray(s2_all, dtype=np.float64)[idx_arr].mean())


def bary_weight_batch(
    M: np.ndarray,
    s2_all: np.ndarray,
    combos: np.ndarray,
    out_Q: np.ndarray,
    out_w: np.ndarray,
) -> None:
    if _cython_bary_weight_batch is not None:
        _cython_bary_weight_batch(M, s2_all, combos, out_Q, out_w)
        return
    print("Warning: no cython module bary_weight_batch. Python fallback.")
    for i, combo in enumerate(combos):
        points = M[combo]
        mean = points.mean(axis=0)
        out_Q[i] = mean
        out_w[i] = float(np.dot(mean, mean) - s2_all[combo].mean())


def union_if_adjacent_int(a: np.ndarray, b: np.ndarray, out_u: np.ndarray) -> bool:
    if _cython_union_if_adjacent_int is not None:
        return bool(_cython_union_if_adjacent_int(a, b, out_u))
    print("Warning: no cython module union_if_adjacent_int. Python fallback.")
    i = j = u = 0
    k = a.shape[0]
    while i < k and j < k:
        if u >= out_u.shape[0]:
            return False
        ai = int(a[i])
        bj = int(b[j])
        if ai == bj:
            out_u[u] = ai
            i += 1
            j += 1
        elif ai < bj:
            out_u[u] = ai
            i += 1
        else:
            out_u[u] = bj
            j += 1
        u += 1
    while i < k:
        if u >= out_u.shape[0]:
            return False
        out_u[u] = int(a[i])
        i += 1
        u += 1
    while j < k:
        if u >= out_u.shape[0]:
            return False
        out_u[u] = int(b[j])
        j += 1
        u += 1
    return u == k + 1


def _minimum_enclosing_ball_fallback(points_sub: np.ndarray) -> tuple[np.ndarray, float]:
    pts = [np.asarray(p, dtype=np.float64) for p in points_sub]
    dim = points_sub.shape[1] if points_sub.size else 0

    def _ball_from(boundary: list[np.ndarray]) -> tuple[np.ndarray, float]:
        if not boundary:
            return np.zeros(dim, dtype=np.float64), -1.0
        arr = np.vstack(boundary)
        if arr.shape[0] == 1:
            return arr[0], 0.0
        if arr.shape[0] == 2:
            diff = arr[0] - arr[1]
            center = 0.5 * (arr[0] + arr[1])
            return center, float(np.dot(diff, diff)) * 0.25
        base = arr[0]
        A = 2.0 * (arr[1:] - base)
        b = np.einsum("ij,ij->i", arr[1:], arr[1:]) - float(np.dot(base, base))
        center = np.linalg.lstsq(A, b, rcond=None)[0] if A.size else base
        radius_sq = float(np.max(np.sum((arr - center) ** 2, axis=1)))
        return center, radius_sq

    rng = np.random.default_rng(0)
    order = list(rng.permutation(len(pts)))
    shuffled = [pts[i] for i in order]

    def _welzl(points: list[np.ndarray], boundary: list[np.ndarray]) -> tuple[np.ndarray, float]:
        if not points or len(boundary) == dim + 1:
            return _ball_from(boundary)
        p = points.pop()
        center, radius_sq = _welzl(points, boundary)
        if radius_sq >= 0 and np.dot(p - center, p - center) <= radius_sq + 1e-9:
            points.append(p)
            return center, radius_sq
        boundary.append(p)
        center, radius_sq = _welzl(points, boundary)
        boundary.pop()
        points.append(p)
        return center, radius_sq

    center, radius_sq = _welzl(shuffled, [])
    if radius_sq < 0:
        radius_sq = 0.0
    return center, radius_sq


def minimum_enclosing_ball(points_sub: np.ndarray) -> tuple[np.ndarray, float]:
    dim = points_sub.shape[1] if points_sub.ndim == 2 else 0
    if points_sub.shape[0] == 0:
        return np.zeros(dim, dtype=np.float64), 0.0
    if points_sub.shape[0] == 1:
        return points_sub[0], 0.0
    if points_sub.shape[0] == 2:
        diff = points_sub[0] - points_sub[1]
        return 0.5 * (points_sub[0] + points_sub[1]), float(np.dot(diff, diff)) * 0.25

    from ._cython import native_minimum_enclosing_ball
    if points_sub.dtype != np.float64:
        points_sub = points_sub.astype(np.float64)
    if not points_sub.flags.c_contiguous:
        points_sub = np.ascontiguousarray(points_sub)
    return native_minimum_enclosing_ball(points_sub)

def kth_radius(M: np.ndarray, k: int, metric: str, precomputed: bool) -> np.ndarray:
    if precomputed:
        return np.partition(M, k, axis=1)[:, k]
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(M)
    dists, _ = nn.kneighbors(M)
    return dists[:, k]
