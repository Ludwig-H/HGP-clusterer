# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
"""Cython utilities for HypergraphPercol."""

import numpy as np
cimport numpy as np


cdef class UnionFind:
    cdef np.intp_t[:] parent
    cdef np.intp_t[:] _size

    def __init__(self, int n):
        self.parent = np.arange(n, dtype=np.intp)
        self._size = np.ones(n, dtype=np.intp)

    cpdef int find(self, int x):
        cdef int r = x
        while self.parent[r] != r:
            r = self.parent[r]
        cdef int cur = x
        cdef int nxt
        while self.parent[cur] != r:
            nxt = self.parent[cur]
            self.parent[cur] = r
            cur = nxt
        return r

    cpdef bint union(self, int x, int y):
        cdef int rx = self.find(x)
        cdef int ry = self.find(y)
        if rx == ry:
            return False
        if self._size[rx] < self._size[ry]:
            self.parent[rx] = ry
            self._size[ry] += self._size[rx]
        else:
            self.parent[ry] = rx
            self._size[rx] += self._size[ry]
        return True

    cpdef int component_size(self, int x):
        return self._size[self.find(x)]


ctypedef np.double_t DTYPE_t
ctypedef np.int64_t ITYPE_t


def kruskal(U, V, W, int N):
    """
    Kruskal sans tri (W déjà trié par ordre croissant).

    Entrée:
      - U, V: arrays d'entiers (0..N-1), U[i] < V[i]
      - W:    array de flottants (poids), déjà trié croissant
      - N:    nombre de sommets

    Sortie:
      - Une liste de ndarrays d'indices d'arêtes (dtype=np.intp), un par composante.
        Les nœuds isolés donnent un tableau vide. Si le graphe est connexe: liste de taille 1.
    """
    cdef Py_ssize_t M
    cdef Py_ssize_t i, e
    cdef int a, b
    cdef int components = N
    cdef int r
    cdef np.intp_t C, c

    # Contiguïté + dtypes internes
    U = np.ascontiguousarray(U, dtype=np.intp)
    V = np.ascontiguousarray(V, dtype=np.intp)
    W = np.ascontiguousarray(W, dtype=np.float64)

    M = (<np.ndarray> U).shape[0]
    if (<np.ndarray> V).shape[0] != M or (<np.ndarray> W).shape[0] != M:
        raise ValueError("U, V et W doivent avoir la même longueur")

    cdef np.intp_t[:] Uv = U
    cdef np.intp_t[:] Vv = V
    # On ne touche pas à W ici, les arêtes sont déjà triées

    cdef UnionFind uf = UnionFind(N)

    # Indices des arêtes retenues (buffer max M)
    cdef np.ndarray[np.intp_t, ndim=1] idx_mst = np.empty(M, dtype=np.intp)
    cdef np.intp_t[:] idx_mstv = idx_mst
    cdef Py_ssize_t k = 0

    # Boucle principale de Kruskal
    for i in range(M):
        a = <int> Uv[i]
        b = <int> Vv[i]
        if uf.union(a, b):
            idx_mstv[k] = <np.intp_t> i
            k += 1
            components -= 1
            if components == 1:  # arrêt anticipé si connexe
                break

    # --- Regroupement par composante: schéma 2 passes, sans dict ---

    # 1) Racine de chaque sommet
    cdef np.ndarray[np.intp_t, ndim=1] roots_arr = np.empty(N, dtype=np.intp)
    cdef np.intp_t[:] roots = roots_arr
    for i in range(N):
        roots[i] = uf.find(<int> i)

    # 2) Compactage racine -> id de composante 0..C-1 (root_to_cc), init à -1
    cdef np.ndarray[np.intp_t, ndim=1] root_to_cc = np.empty(N, dtype=np.intp)
    cdef np.intp_t[:] r2c = root_to_cc
    for i in range(N):
        r2c[i] = -1

    C = 0
    for i in range(N):
        r = <int> roots[i]
        if r2c[r] == -1:
            r2c[r] = C
            C += 1

    # 3) Compter le nb d'arêtes MST par composante
    cdef np.ndarray[np.intp_t, ndim=1] counts = np.zeros(C, dtype=np.intp)
    cdef np.intp_t[:] cnt = counts
    for i in range(k):
        e = idx_mstv[i]
        r = <int> roots[ Uv[e] ]  # U[e] et V[e] ont la même racine dans le MST
        cnt[ r2c[r] ] += 1

    # 4) Allouer les sorties et offsets
    cdef list out = [None] * C
    cdef np.ndarray[np.intp_t, ndim=1] offsets = np.zeros(C, dtype=np.intp)
    cdef np.intp_t[:] off = offsets

    cdef np.ndarray[np.intp_t, ndim=1] arr
    cdef np.intp_t[:] arr_view

    for i in range(C):
        if cnt[i] == 0:
            out[i] = np.empty(0, dtype=np.intp)
        else:
            out[i] = np.empty(cnt[i], dtype=np.intp)

    # 5) Remplissage des indices par composante
    for i in range(k):
        e = idx_mstv[i]
        r = <int> roots[ Uv[e] ]
        c = r2c[r]
        arr = <np.ndarray[np.intp_t, ndim=1]> out[c]
        arr_view = arr  # <- conversion propre en memoryview
        arr_view[ off[c] ] = <np.intp_t> e
        off[c] += 1

    return out



cpdef double bary_weight_one(
    DTYPE_t[:, ::1] M,
    DTYPE_t[::1] s2_all,
    ITYPE_t[::1] idx,
    DTYPE_t[::1] out_q,
):
    cdef Py_ssize_t k = idx.shape[0]
    cdef Py_ssize_t d = M.shape[1]
    cdef Py_ssize_t i, t
    cdef double smean = 0.0
    cdef double qnorm2 = 0.0
    cdef ITYPE_t ii

    for t in range(d):
        out_q[t] = 0.0

    for i in range(k):
        ii = idx[i]
        smean += s2_all[ii]
        for t in range(d):
            out_q[t] += M[ii, t]

    for t in range(d):
        out_q[t] /= k
        qnorm2 += out_q[t] * out_q[t]

    smean /= k
    return qnorm2 - smean


cpdef void bary_weight_batch(
    DTYPE_t[:, ::1] M,
    DTYPE_t[::1] s2_all,
    ITYPE_t[:, ::1] combos,
    DTYPE_t[:, ::1] out_Q,
    DTYPE_t[::1] out_w,
):
    cdef Py_ssize_t m = combos.shape[0]
    cdef Py_ssize_t k = combos.shape[1]
    cdef Py_ssize_t d = M.shape[1]
    cdef Py_ssize_t i, j, t
    cdef double smean, qnorm2
    cdef ITYPE_t ii

    for i in range(m):
        smean = 0.0
        for t in range(d):
            out_Q[i, t] = 0.0
        for j in range(k):
            ii = combos[i, j]
            smean += s2_all[ii]
            for t in range(d):
                out_Q[i, t] += M[ii, t]
        for t in range(d):
            out_Q[i, t] /= k
        smean /= k
        qnorm2 = 0.0
        for t in range(d):
            qnorm2 += out_Q[i, t] * out_Q[i, t]
        out_w[i] = qnorm2 - smean


cpdef int union_if_adjacent_int(
    ITYPE_t[::1] a,
    ITYPE_t[::1] b,
    ITYPE_t[::1] out_u,
):
    cdef Py_ssize_t k = a.shape[0]
    cdef Py_ssize_t i = 0
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t u = 0

    while i < k and j < k:
        if u >= out_u.shape[0]:
            return 0
        if a[i] == b[j]:
            out_u[u] = a[i]
            i += 1
            j += 1
            u += 1
        elif a[i] < b[j]:
            out_u[u] = a[i]
            i += 1
            u += 1
        else:
            out_u[u] = b[j]
            j += 1
            u += 1

    while i < k:
        if u >= out_u.shape[0]:
            return 0
        out_u[u] = a[i]
        i += 1
        u += 1

    while j < k:
        if u >= out_u.shape[0]:
            return 0
        out_u[u] = b[j]
        j += 1
        u += 1

    return 1 if u == k + 1 else 0


cdef inline np.int64_t _min_i64(np.int64_t a, np.int64_t b) nogil:
    return a if a <= b else b


cdef inline np.int64_t _max_i64(np.int64_t a, np.int64_t b) nogil:
    return a if a >= b else b


cpdef tuple build_leaf_dfs_intervals(
    np.ndarray[np.int64_t, ndim=1] left,
    np.ndarray[np.int64_t, ndim=1] right,
):
    cdef Py_ssize_t t = left.shape[0]
    if right.shape[0] != t:
        raise ValueError("left/right must have same length")
    cdef Py_ssize_t m = t + 1
    cdef Py_ssize_t n_nodes = m + t

    cdef np.int64_t[:] L = left
    cdef np.int64_t[:] R = right

    cdef np.ndarray[np.int64_t, ndim=1] first = np.empty(n_nodes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] last = np.empty(n_nodes, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] leaf_order = np.empty(m, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] pos = np.empty(m, dtype=np.int64)

    cdef np.int64_t[:] first_v = first
    cdef np.int64_t[:] last_v = last
    cdef np.int64_t[:] lo_v = leaf_order
    cdef np.int64_t[:] pos_v = pos

    cdef Py_ssize_t i
    for i in range(n_nodes):
        first_v[i] = -1
        last_v[i] = -1

    cdef np.ndarray[np.int64_t, ndim=1] stack_node = np.empty(n_nodes, dtype=np.int64)
    cdef np.ndarray[np.int8_t, ndim=1] stack_st = np.empty(n_nodes, dtype=np.int8)
    cdef np.int64_t[:] st_node = stack_node
    cdef np.int8_t[:] st_st = stack_st

    cdef Py_ssize_t sp = 0
    cdef np.int64_t root = m + t - 1
    st_node[sp] = root
    st_st[sp] = 0
    sp += 1

    cdef Py_ssize_t k = 0
    cdef np.int64_t x, state, child_idx, a, b, fa, fb, la, lb

    while sp > 0:
        sp -= 1
        x = st_node[sp]
        state = st_st[sp]

        if x < m:
            first_v[x] = k
            last_v[x] = k
            lo_v[k] = x
            k += 1
            continue

        child_idx = x - m
        if not (0 <= child_idx < t):
            raise ValueError("Invalid internal node index")

        if state == 0:
            st_node[sp] = x
            st_st[sp] = 1
            sp += 1
            b = R[child_idx]
            a = L[child_idx]
            if a >= x or b >= x or a < 0 or b < 0:
                raise ValueError("SciPy linkage convention violated: child >= parent")
            st_node[sp] = b
            st_st[sp] = 0
            sp += 1
            st_node[sp] = a
            st_st[sp] = 0
            sp += 1
        else:
            a = L[child_idx]
            b = R[child_idx]
            fa = first_v[a]
            fb = first_v[b]
            la = last_v[a]
            lb = last_v[b]
            if fa == -1 or fb == -1:
                raise ValueError("Invalid tree: child interval not computed")
            first_v[x] = _min_i64(fa, fb)
            last_v[x] = _max_i64(la, lb)

    if k != m:
        raise ValueError("Leaf DFS did not visit all leaves")

    for i in range(m):
        pos_v[lo_v[i]] = i

    return pos, first, last, leaf_order
