from __future__ import annotations

import numpy as np

try:
    import cyminiball as _CYMINIBALL  # type: ignore
except Exception as exc:  # pragma: no cover - hard failure, dependency required
    raise ImportError("cyminiball is required for minimum_enclosing_ball") from exc

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


def minimum_enclosing_ball(points_sub: np.ndarray) -> tuple[np.ndarray, float]:
    if points_sub.shape[0] <= 1:
        return points_sub[0], 0.0
    if points_sub.shape[0] == 2:
        diff = points_sub[0] - points_sub[1]
        return 0.5 * (points_sub[0] + points_sub[1]), float(np.dot(diff, diff)) * 0.25

    ball = _CYMINIBALL.Miniball(points_sub)
    center = np.asarray(ball.center(), dtype=np.float64)
    radius_sq = float(ball.squared_radius())
    return center, radius_sq


def kth_radius(M: np.ndarray, k: int, metric: str, precomputed: bool) -> np.ndarray:
    if precomputed:
        return np.partition(M, k, axis=1)[:, k]
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(M)
    dists, _ = nn.kneighbors(M)
    return dists[:, k]
