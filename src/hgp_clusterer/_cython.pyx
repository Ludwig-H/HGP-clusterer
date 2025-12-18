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


from libcpp.vector cimport vector
from libc.math cimport NAN, isnan, fabs

def condense_tree_cython(
    DTYPE_t[::1] W_nodes,
    ITYPE_t[::1] U_mst,
    ITYPE_t[::1] V_mst,
    DTYPE_t[::1] W_mst,
    ITYPE_t min_cluster_size,
    bint check_sorted=True
):
    """
    Cython optimized version of condense_tree.
    """
    cdef Py_ssize_t N = W_nodes.shape[0]
    cdef Py_ssize_t M = W_mst.shape[0]
    cdef Py_ssize_t i
    
    # Declarations for inside loop
    cdef vector[ITYPE_t] new_ch
    cdef vector[ITYPE_t] new_edges

    # Internal Union-Find state (we don't use the class to avoid overhead/api mismatch)
    cdef vector[ITYPE_t] parent = vector[ITYPE_t](N)
    cdef vector[double] comp_weight = vector[double](N)
    cdef vector[ITYPE_t] comp_cid = vector[ITYPE_t](N)
    cdef vector[vector[ITYPE_t]] comp_edges = vector[vector[ITYPE_t]](N)

    # Cluster data
    cdef vector[vector[ITYPE_t]] children
    cdef vector[double] birth_r
    cdef vector[double] death_r
    cdef vector[double] stability
    cdef vector[vector[ITYPE_t]] cluster_edges
    cdef vector[double] size_at_birth
    cdef vector[double] n_in_cluster
    cdef vector[double] sum_join_lambda

    cdef double EPS = 1e-12
    cdef ITYPE_t u, v, ru, rv, cid, cid_u, cid_v, cid_new
    cdef double r, lam, n_in, n_parent

    if U_mst.shape[0] != M or V_mst.shape[0] != M:
        raise ValueError("U_mst, V_mst, W_mst must have same length M")
    if N != M + 1:
        raise ValueError(f"Expected N = M + 1, got N={N}, M={M}")
    
    if check_sorted:
        for i in range(M - 1):
            if W_mst[i+1] < W_mst[i]:
                raise ValueError("W_mst must be sorted in non-decreasing order")
    
    for i in range(N):
        parent[i] = i
        comp_weight[i] = W_nodes[i]
        comp_cid[i] = -1

    for i in range(M):
        u = U_mst[i]
        v = V_mst[i]
        r = W_mst[i]
        lam = 1.0 / (r + EPS)

        # Find with path compression
        ru = u
        while parent[ru] != ru:
            parent[ru] = parent[parent[ru]]
            ru = parent[ru]
        
        rv = v
        while parent[rv] != rv:
            parent[rv] = parent[parent[rv]]
            rv = parent[rv]
        
        if ru == rv:
            continue
            
        # elig_u/elig_v based on comp_cid != -1
        # Python: elig_u = comp_cid[ru] != -1
        
        if comp_cid[ru] == -1 and comp_cid[rv] == -1:
            # Case 1: Both ineligible
            if comp_weight[ru] < comp_weight[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            comp_weight[ru] += comp_weight[rv]
            
            # Merge edges
            if not comp_edges[rv].empty():
                comp_edges[ru].insert(comp_edges[ru].end(), comp_edges[rv].begin(), comp_edges[rv].end())
                comp_edges[rv].clear()
            comp_edges[ru].push_back(i)
            
            # Check if becomes eligible
            if comp_cid[ru] == -1 and comp_weight[ru] >= min_cluster_size:
                # New leaf
                cid = children.size()
                children.push_back(vector[ITYPE_t]())
                birth_r.push_back(r)
                death_r.push_back(NAN)
                stability.push_back(0.0)
                cluster_edges.push_back(comp_edges[ru]) # Copy
                comp_edges[ru].clear()
                
                size_at_birth.push_back(comp_weight[ru])
                n_in_cluster.push_back(comp_weight[ru])
                sum_join_lambda.push_back(comp_weight[ru] * lam)
                
                comp_cid[ru] = cid

        elif comp_cid[ru] != -1 and comp_cid[rv] == -1:
            # Case 2: ru eligible, rv ineligible -> attach rv to ru
            parent[rv] = ru
            comp_weight[ru] += comp_weight[rv]
            cid = comp_cid[ru]
            
            if not comp_edges[rv].empty():
                cluster_edges[cid].insert(cluster_edges[cid].end(), comp_edges[rv].begin(), comp_edges[rv].end())
                comp_edges[rv].clear()
            cluster_edges[cid].push_back(i)
            
            n_in = comp_weight[rv]
            n_in_cluster[cid] += n_in
            sum_join_lambda[cid] += n_in * lam

        elif comp_cid[ru] == -1 and comp_cid[rv] != -1:
            # Case 3: ru ineligible, rv eligible -> attach ru to rv
            parent[ru] = rv
            comp_weight[rv] += comp_weight[ru]
            cid = comp_cid[rv]
            
            if not comp_edges[ru].empty():
                cluster_edges[cid].insert(cluster_edges[cid].end(), comp_edges[ru].begin(), comp_edges[ru].end())
                comp_edges[ru].clear()
            cluster_edges[cid].push_back(i)
            
            n_in = comp_weight[ru]
            n_in_cluster[cid] += n_in
            sum_join_lambda[cid] += n_in * lam
            
        else:
            # Case 4: Both eligible -> Merge two clusters
            cid_u = comp_cid[ru]
            cid_v = comp_cid[rv]
            
            # Close children
            if isnan(death_r[cid_u]):
                death_r[cid_u] = r
                stability[cid_u] += sum_join_lambda[cid_u] - n_in_cluster[cid_u] * lam
            
            if isnan(death_r[cid_v]):
                death_r[cid_v] = r
                stability[cid_v] += sum_join_lambda[cid_v] - n_in_cluster[cid_v] * lam
            
            # Create parent
            cid_new = children.size()
            new_ch.clear() # Clear reused vector
            new_ch.push_back(cid_u)
            new_ch.push_back(cid_v)
            children.push_back(new_ch)
            
            birth_r.push_back(r)
            death_r.push_back(NAN)
            stability.push_back(0.0)
            
            # Merge edges of children + current edge
            new_edges.clear() # Clear reused vector
            if not cluster_edges[cid_u].empty():
                new_edges.insert(new_edges.end(), cluster_edges[cid_u].begin(), cluster_edges[cid_u].end())
            if not cluster_edges[cid_v].empty():
                new_edges.insert(new_edges.end(), cluster_edges[cid_v].begin(), cluster_edges[cid_v].end())
            new_edges.push_back(i)
            cluster_edges.push_back(new_edges)
            
            n_parent = n_in_cluster[cid_u] + n_in_cluster[cid_v]
            size_at_birth.push_back(n_parent)
            n_in_cluster.push_back(n_parent)
            sum_join_lambda.push_back(n_parent * lam)
            
            # Union components
            if comp_weight[ru] < comp_weight[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            comp_weight[ru] += comp_weight[rv]
            comp_cid[ru] = cid_new
            comp_cid[rv] = -1

    # Finalize stability
    cdef Py_ssize_t n_clusters = children.size()
    cdef np.ndarray[np.float64_t, ndim=1] lambda_birth_arr = np.empty(n_clusters, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] lambda_death_arr = np.empty(n_clusters, dtype=np.float64)
    
    for i in range(n_clusters):
        lambda_birth_arr[i] = 1.0 / (birth_r[i] + EPS)
        if isnan(death_r[i]):
            stability[i] += sum_join_lambda[i]
            lambda_death_arr[i] = 0.0
        else:
            lambda_death_arr[i] = 1.0 / (death_r[i] + EPS)
            
    # Convert vectors to Python objects for return
    py_children = []
    for i in range(n_clusters):
        py_children.append(children[i])
        
    py_cluster_edges = []
    for i in range(n_clusters):
        py_cluster_edges.append(cluster_edges[i])
        
    return {
        'children': py_children,
        'r': np.asarray(birth_r, dtype=np.float64),
        'stability': np.asarray(stability, dtype=np.float64),
        'edges': py_cluster_edges,
        'size': np.asarray(size_at_birth, dtype=np.float64),
        'lambda_birth': lambda_birth_arr,
        'lambda_death': lambda_death_arr,
        'U': np.asarray(U_mst),
        'V': np.asarray(V_mst),
        'W': np.asarray(W_mst),
        'N': int(N),
        'M': int(M),
    }


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
