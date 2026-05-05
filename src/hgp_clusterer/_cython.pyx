# cython: language_level=3
# cython: boundscheck=False
# cython: nonecheck=False
# cython: initializedcheck=False
"""Cython utilities for HypergraphPercol."""

import numpy as np
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.math cimport NAN, isnan, fabs
from cython.parallel import prange

# -----------------------------------------------------------------------------
# 1. Type Definitions (32-bit Optimization)
# -----------------------------------------------------------------------------
ctypedef np.float32_t DTYPE_t
ctypedef np.int32_t ITYPE_t

# -----------------------------------------------------------------------------
# 2. C++ External Definitions
# -----------------------------------------------------------------------------
cdef extern from *:
    """
    #include <vector>
    #include <algorithm>
    
    struct FaceRef {
        int simplex_idx;
        int drop_idx;
    };

    struct FaceRefComparator {
        const int* simplex_indices; // Pointer to the flat simplex array (N * (K+1))
        int K_plus_1;               // Stride (K+1)

        FaceRefComparator() : simplex_indices(nullptr), K_plus_1(0) {}
        FaceRefComparator(const int* indices, int k_plus_1) 
            : simplex_indices(indices), K_plus_1(k_plus_1) {}

        bool operator()(const FaceRef& a, const FaceRef& b) const {
            const int* sim_a = simplex_indices + (size_t)a.simplex_idx * K_plus_1;
            const int* sim_b = simplex_indices + (size_t)b.simplex_idx * K_plus_1;

            int ia = 0; 
            int ib = 0; 
            
            // Lexicographical comparison of the K vertices (skipping drop_idx)
            for (int k = 0; k < K_plus_1 - 1; ++k) {
                if (ia == a.drop_idx) ia++;
                if (ib == b.drop_idx) ib++;
                
                int val_a = sim_a[ia];
                int val_b = sim_b[ib];
                
                if (val_a != val_b) {
                    return val_a < val_b;
                }
                ia++;
                ib++;
            }
            return false; 
        }
    };
    """
    cdef struct FaceRef:
        int simplex_idx
        int drop_idx
    
    cdef cppclass FaceRefComparator:
        FaceRefComparator()
        FaceRefComparator(const int* indices, int k_plus_1)
        bool operator()(const FaceRef& a, const FaceRef& b) nogil

cdef extern from "_miniball_batch.hpp":
    void compute_miniball_radii_batch(
        const double* M,
        const int* simplex_indices,
        const unsigned char* mask,
        double* radii,
        size_t n_simplices,
        size_t K_plus_1,
        size_t dim
    ) nogil

    void compute_single_miniball(
        const double* points_flat,
        size_t n_points,
        size_t dim,
        double* out_center,
        double* out_radius_sq
    ) nogil

def native_minimum_enclosing_ball(const double[:, ::1] points):
    """
    Computes minimum enclosing ball natively for a single simplex.
    """
    cdef size_t n_points = points.shape[0]
    cdef size_t dim = points.shape[1]
    
    center = np.zeros(dim, dtype=np.float64)
    cdef double[::1] center_view = center
    cdef double radius_sq = 0.0
    
    with nogil:
        compute_single_miniball(
            &points[0, 0],
            n_points,
            dim,
            &center_view[0],
            &radius_sq
        )
        
    return center, radius_sq

def compute_fallback_radii(
    const double[:, ::1] M,
    const int[:, ::1] simplex_indices,
    const unsigned char[::1] mask,
    double[::1] radii
):
    cdef size_t n_simplices = simplex_indices.shape[0]
    cdef size_t K_plus_1 = simplex_indices.shape[1]
    cdef size_t dim = M.shape[1]
    
    with nogil:
        compute_miniball_radii_batch(
            &M[0, 0],
            &simplex_indices[0, 0],
            &mask[0],
            &radii[0],
            n_simplices,
            K_plus_1,
            dim
        )

    
cdef extern from *:
    void std_sort_custom "std::sort" [Iter, Compare](Iter first, Iter last, Compare comp) nogil

cdef class UnionFind:
    cdef ITYPE_t[:] parent
    cdef ITYPE_t[:] _size

    def __init__(self, int n):
        self.parent = np.arange(n, dtype=np.int32)
        self._size = np.ones(n, dtype=np.int32)

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


def kruskal(U, V, W, int N):
    """
    Kruskal sans tri (W déjà trié par ordre croissant).

    Entrée:
      - U, V: arrays d'entiers (0..N-1), U[i] < V[i] (int32)
      - W:    array de flottants (poids), déjà trié croissant (float32)
      - N:    nombre de sommets

    Sortie:
      - Une liste de ndarrays d'indices d'arêtes (dtype=np.int32), un par composante.
    """
    cdef Py_ssize_t M
    cdef Py_ssize_t i, e
    cdef int a, b
    cdef int components = N
    cdef int r
    cdef int C, c

    # Contiguïté + dtypes internes (Force 32-bit)
    U = np.ascontiguousarray(U, dtype=np.int32)
    V = np.ascontiguousarray(V, dtype=np.int32)
    W = np.ascontiguousarray(W, dtype=np.float32)

    M = (<np.ndarray> U).shape[0]
    if (<np.ndarray> V).shape[0] != M or (<np.ndarray> W).shape[0] != M:
        raise ValueError("U, V et W doivent avoir la même longueur")

    cdef ITYPE_t[:] Uv = U
    cdef ITYPE_t[:] Vv = V
    # On ne touche pas à W ici, les arêtes sont déjà triées

    cdef UnionFind uf = UnionFind(N)

    # Indices des arêtes retenues (buffer max M)
    cdef np.ndarray[ITYPE_t, ndim=1] idx_mst = np.empty(M, dtype=np.int32)
    cdef ITYPE_t[:] idx_mstv = idx_mst
    cdef Py_ssize_t k = 0

    # Boucle principale de Kruskal
    for i in range(M):
        a = <int> Uv[i]
        b = <int> Vv[i]
        if uf.union(a, b):
            idx_mstv[k] = <ITYPE_t> i
            k += 1
            components -= 1
            if components == 1:  # arrêt anticipé si connexe
                break

    # --- Regroupement par composante: schéma 2 passes, sans dict ---

    # 1) Racine de chaque sommet
    cdef np.ndarray[ITYPE_t, ndim=1] roots_arr = np.empty(N, dtype=np.int32)
    cdef ITYPE_t[:] roots = roots_arr
    for i in range(N):
        roots[i] = uf.find(<int> i)

    # 2) Compactage racine -> id de composante 0..C-1 (root_to_cc), init à -1
    cdef np.ndarray[ITYPE_t, ndim=1] root_to_cc = np.empty(N, dtype=np.int32)
    cdef ITYPE_t[:] r2c = root_to_cc
    for i in range(N):
        r2c[i] = -1

    C = 0
    for i in range(N):
        r = <int> roots[i]
        if r2c[r] == -1:
            r2c[r] = C
            C += 1

    # 3) Compter le nb d'arêtes MST par composante
    cdef np.ndarray[ITYPE_t, ndim=1] counts = np.zeros(C, dtype=np.int32)
    cdef ITYPE_t[:] cnt = counts
    for i in range(k):
        e = idx_mstv[i]
        r = <int> roots[ Uv[e] ]  # U[e] et V[e] ont la même racine dans le MST
        cnt[ r2c[r] ] += 1

    # 4) Allouer les sorties et offsets
    cdef list out = [None] * C
    cdef np.ndarray[ITYPE_t, ndim=1] offsets = np.zeros(C, dtype=np.int32)
    cdef ITYPE_t[:] off = offsets

    cdef np.ndarray[ITYPE_t, ndim=1] arr
    cdef ITYPE_t[:] arr_view

    for i in range(C):
        if cnt[i] == 0:
            out[i] = np.empty(0, dtype=np.int32)
        else:
            out[i] = np.empty(cnt[i], dtype=np.int32)

    # 5) Remplissage des indices par composante
    for i in range(k):
        e = idx_mstv[i]
        r = <int> roots[ Uv[e] ]
        c = r2c[r]
        arr = <np.ndarray[ITYPE_t, ndim=1]> out[c]
        arr_view = arr  # <- conversion propre en memoryview
        arr_view[ off[c] ] = <ITYPE_t> e
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


cdef inline ITYPE_t _min_i32(ITYPE_t a, ITYPE_t b) nogil:
    return a if a <= b else b


cdef inline ITYPE_t _max_i32(ITYPE_t a, ITYPE_t b) nogil:
    return a if a >= b else b


def condense_tree_cython(
    DTYPE_t[::1] W_nodes,
    ITYPE_t[::1] U_mst,
    ITYPE_t[::1] V_mst,
    DTYPE_t[::1] W_mst,
    ITYPE_t min_cluster_size,
    bint check_sorted=True,
    double epsilon=0.0
):
    """
    Cython optimized version of condense_tree with N-ary support via epsilon.
    Uses float32 and int32 for memory efficiency.
    """
    cdef Py_ssize_t N = W_nodes.shape[0]
    cdef Py_ssize_t M = W_mst.shape[0]
    cdef Py_ssize_t i, j, k
    
    # Output structures
    cdef vector[vector[ITYPE_t]] children
    cdef vector[float] birth_r
    cdef vector[float] death_r
    cdef vector[float] stability
    cdef vector[float] size_at_birth
    cdef vector[float] n_in_cluster
    cdef vector[float] sum_join_lambda
    
    # Internal state
    cdef vector[ITYPE_t] parent = vector[ITYPE_t](N)
    cdef vector[float] comp_weight = vector[float](N)
    cdef vector[ITYPE_t] comp_cid = vector[ITYPE_t](N)
    cdef vector[vector[ITYPE_t]] comp_nodes = vector[vector[ITYPE_t]](N)
    
    # To track which clusters are merging into a component during a batch
    cdef vector[vector[ITYPE_t]] tracked_cids = vector[vector[ITYPE_t]](N)

    # Map each point to the FIRST cluster (leaf) it enters.
    cdef np.ndarray[ITYPE_t, ndim=1] initial_membership = np.full(N, -1, dtype=np.int32)
    # Track the distance r at which each point joins its cluster
    cdef np.ndarray[np.float32_t, ndim=1] join_r = np.full(N, np.inf, dtype=np.float32)

    cdef double EPS = 1e-12
    cdef ITYPE_t u, v, ru, rv, cid, node_idx
    cdef float r, lam, n_in, w_start, w_current, added_weight, n_parent
    cdef Py_ssize_t j_node, c_idx, cid_new
    cdef vector[ITYPE_t] new_ch
    cdef bint has_clusters

    if U_mst.shape[0] != M or V_mst.shape[0] != M:
        raise ValueError("U_mst, V_mst, W_mst must have same length M")
    if N != M + 1:
        raise ValueError(f"Expected N = M + 1, got N={N}, M={M}")
    
    if check_sorted:
        for i in range(M - 1):
            if W_mst[i+1] < W_mst[i]:
                raise ValueError("W_mst must be sorted in non-decreasing order")
    
    # Initialization
    for i in range(N):
        parent[i] = i
        comp_weight[i] = W_nodes[i]
        comp_cid[i] = -1
        comp_nodes[i].push_back(i)

    # We need to allocate `last_seen_batch` once outside.
    cdef vector[Py_ssize_t] last_seen_batch = vector[Py_ssize_t](N, -1)
    cdef vector[ITYPE_t] unique_roots
    
    # Reset loop index
    i = 0
    while i < M:
        w_start = W_mst[i]
        
        # 1. Identify Batch
        j = i
        while j < M:
            w_current = W_mst[j]
            if w_current - w_start <= epsilon:
                j += 1
            else:
                break
        
        # Lam for this batch
        r = W_mst[j-1]
        lam = 1.0 / (r + EPS)
        
        unique_roots.clear()
        
        # 2. Process Merges
        for k in range(i, j):
            u = U_mst[k]
            v = V_mst[k]
            
            ru = u
            while parent[ru] != ru: parent[ru] = parent[parent[ru]]; ru = parent[ru]
            rv = v
            while parent[rv] != rv: parent[rv] = parent[parent[rv]]; rv = parent[rv]
            
            if ru == rv:
                continue
                
            if comp_weight[ru] < comp_weight[rv]:
                ru, rv = rv, ru
            
            # Collect children
            if comp_cid[ru] != -1:
                tracked_cids[ru].push_back(comp_cid[ru])
                comp_cid[ru] = -1
            if comp_cid[rv] != -1:
                tracked_cids[rv].push_back(comp_cid[rv])
                comp_cid[rv] = -1
            
            if not tracked_cids[rv].empty():
                tracked_cids[ru].insert(tracked_cids[ru].end(), tracked_cids[rv].begin(), tracked_cids[rv].end())
                tracked_cids[rv].clear()
            
            # Physical Union
            parent[rv] = ru
            comp_weight[ru] += comp_weight[rv]
            comp_nodes[ru].insert(comp_nodes[ru].end(), comp_nodes[rv].begin(), comp_nodes[rv].end())
            comp_nodes[rv].clear()
            
            # Track Root
            if last_seen_batch[ru] != i:
                unique_roots.push_back(ru)
                last_seen_batch[ru] = i
                
        # 3. Analyze Roots
        for k in range(unique_roots.size()):
            ru = unique_roots[k]
            
            # It might have been merged into another root subsequent to being added to unique_roots
            # Verify if it is still a root
            if parent[ru] != ru:
                continue
                
            # Logic for Cluster Creation
            has_clusters = not tracked_cids[ru].empty()
            
            if comp_weight[ru] >= min_cluster_size:
                if not has_clusters:
                    # Case: New Leaf created from noise
                    cid = children.size()
                    children.push_back(vector[ITYPE_t]()) # No children
                    birth_r.push_back(r)
                    death_r.push_back(NAN)
                    stability.push_back(0.0)
                    size_at_birth.push_back(comp_weight[ru])
                    n_in_cluster.push_back(comp_weight[ru])
                    sum_join_lambda.push_back(comp_weight[ru] * lam)
                    
                    # Assign membership
                    for j_node in range(comp_nodes[ru].size()):
                        node_idx = comp_nodes[ru][j_node]
                        initial_membership[node_idx] = cid
                        join_r[node_idx] = r
                    comp_nodes[ru].clear()
                    
                    comp_cid[ru] = cid
                    
                else:
                    # Case: We have children (merged clusters)
                    # Count how many children
                    if tracked_cids[ru].size() == 1:
                        # Merged 1 cluster with noise -> Just Extend the cluster
                        cid = tracked_cids[ru][0]
                        
                        n_in = comp_weight[ru] # Total weight now
                        
                        added_weight = 0.0
                        for j_node in range(comp_nodes[ru].size()):
                            node_idx = comp_nodes[ru][j_node]
                            initial_membership[node_idx] = cid
                            join_r[node_idx] = r
                            added_weight += W_nodes[node_idx]
                        comp_nodes[ru].clear()
                        
                        n_in_cluster[cid] += added_weight
                        sum_join_lambda[cid] += added_weight * lam
                        
                        # Restore comp_cid
                        comp_cid[ru] = cid
                        
                    else:
                        # Merged >= 2 clusters
                        # Create NEW PARENT
                        new_ch.clear()
                        new_ch = tracked_cids[ru] # Copy
                        
                        # Update children's death
                        n_parent = 0.0
                        for c_idx in range(new_ch.size()):
                            cid = new_ch[c_idx]
                            if isnan(death_r[cid]):
                                death_r[cid] = r
                                stability[cid] += sum_join_lambda[cid] - n_in_cluster[cid] * lam
                            n_parent += n_in_cluster[cid]
                        
                        # Add new noise to n_parent
                        added_weight = 0.0
                        
                        cid_new = children.size()
                        children.push_back(new_ch)
                        birth_r.push_back(r)
                        death_r.push_back(NAN)
                        stability.push_back(0.0)
                        
                        for j_node in range(comp_nodes[ru].size()):
                            node_idx = comp_nodes[ru][j_node]
                            initial_membership[node_idx] = cid_new
                            join_r[node_idx] = r
                            added_weight += W_nodes[node_idx]
                        comp_nodes[ru].clear()
                        
                        n_parent += added_weight
                        size_at_birth.push_back(n_parent)
                        n_in_cluster.push_back(n_parent)
                        sum_join_lambda.push_back(n_parent * lam)
                        
                        comp_cid[ru] = cid_new
            
            # Clear tracked_cids for next usage
            tracked_cids[ru].clear()
            
        # Advance i
        i = j


    # Finalize stability
    cdef Py_ssize_t n_clusters = children.size()
    cdef np.ndarray[np.float32_t, ndim=1] lambda_birth_arr = np.empty(n_clusters, dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] lambda_death_arr = np.empty(n_clusters, dtype=np.float32)
    
    for i in range(n_clusters):
        lambda_birth_arr[i] = 1.0 / (birth_r[i] + EPS)
        if isnan(death_r[i]):
            stability[i] += sum_join_lambda[i]
            lambda_death_arr[i] = 0.0
        else:
            lambda_death_arr[i] = 1.0 / (death_r[i] + EPS)
            
    # Convert vectors to Python objects
    py_children = []
    for i in range(n_clusters):
        py_children.append(children[i])
        
    return {
        'children': py_children,
        'r': np.asarray(birth_r, dtype=np.float32),
        'stability': np.asarray(stability, dtype=np.float32),
        'initial_membership': initial_membership,
        'join_r': join_r,
        'size': np.asarray(size_at_birth, dtype=np.float32),
        'lambda_birth': lambda_birth_arr,
        'lambda_death': lambda_death_arr,
        'U': np.asarray(U_mst, dtype=np.int32),
        'V': np.asarray(V_mst, dtype=np.int32),
        'W': np.asarray(W_mst, dtype=np.float32),
        'N': int(N),
        'M': int(M),
    }


cpdef tuple build_leaf_dfs_intervals(
    np.ndarray[np.int32_t, ndim=1] left,
    np.ndarray[np.int32_t, ndim=1] right,
):
    cdef Py_ssize_t t = left.shape[0]
    if right.shape[0] != t:
        raise ValueError("left/right must have same length")
    cdef Py_ssize_t m = t + 1
    cdef Py_ssize_t n_nodes = m + t

    cdef ITYPE_t[:] L = left
    cdef ITYPE_t[:] R = right

    cdef np.ndarray[ITYPE_t, ndim=1] first = np.empty(n_nodes, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] last = np.empty(n_nodes, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] leaf_order = np.empty(m, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] pos = np.empty(m, dtype=np.int32)

    cdef ITYPE_t[:] first_v = first
    cdef ITYPE_t[:] last_v = last
    cdef ITYPE_t[:] lo_v = leaf_order
    cdef ITYPE_t[:] pos_v = pos

    cdef Py_ssize_t i
    for i in range(n_nodes):
        first_v[i] = -1
        last_v[i] = -1

    cdef np.ndarray[ITYPE_t, ndim=1] stack_node = np.empty(n_nodes, dtype=np.int32)
    cdef np.ndarray[np.int8_t, ndim=1] stack_st = np.empty(n_nodes, dtype=np.int8)
    cdef ITYPE_t[:] st_node = stack_node
    cdef np.int8_t[:] st_st = stack_st

    cdef Py_ssize_t sp = 0
    cdef int root = m + t - 1
    st_node[sp] = root
    st_st[sp] = 0
    sp += 1

    cdef Py_ssize_t k = 0
    cdef int x, child_idx, a, b, fa, fb, la, lb
    cdef int state

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
            first_v[x] = _min_i32(fa, fb)
            last_v[x] = _max_i32(la, lb)

    if k != m:
        raise ValueError("Leaf DFS did not visit all leaves")

    for i in range(m):
        pos_v[lo_v[i]] = i

    return pos, first, last, leaf_order



def build_dual_graph_cython(
    const int[:, ::1] simplex_indices, # (N, K+1) int32
    const float[::1] simplex_weights,  # (N,) float32
    int K
):
    """
    Builds the dual graph (adjacency between simplices sharing a face) using
    an implicit sort-based approach to minimize memory usage.
    
    Strategy: "Generate -> Sort -> Reduce"
    1. Generate references to all faces (FaceRef).
    2. Sort references using a custom comparator that looks up vertex indices (Zero-Copy).
    3. Reduce sorted list to find unique faces and build the graph.
    
    Complexity: O(N * K * log(NK)) Time, O(NK) Memory.
    """
    cdef Py_ssize_t n_simplexes = simplex_indices.shape[0]
    cdef int K_plus_1 = K + 1
    
    if simplex_indices.shape[1] != K_plus_1:
         pass # Let numpy handle potential mismatches or assume correct from caller

    # 1. Generate References (All K-faces of all Simplices)
    cdef vector[FaceRef] all_faces
    cdef Py_ssize_t total_faces = n_simplexes * K_plus_1
    all_faces.resize(total_faces)
    
    cdef Py_ssize_t i, j
    cdef Py_ssize_t idx = 0
    
    # Access raw pointer for C++ comparator
    cdef const int* indices_ptr = &simplex_indices[0, 0]
    
    # Fill references (Parallel)
    # Each i (simplex) writes K+1 entries to all_faces at known positions.
    with nogil:
        for i in prange(n_simplexes, schedule='static'):
            for j in range(K_plus_1):
                # Calculate flat index explicitly for parallel safety
                idx = i * K_plus_1 + j
                all_faces[idx].simplex_idx = <int>i
                all_faces[idx].drop_idx = <int>j
            
    # 2. Sort References
    cdef FaceRefComparator comp = FaceRefComparator(indices_ptr, K_plus_1)
    
    with nogil:
        std_sort_custom(all_faces.begin(), all_faces.end(), comp)

    # 3. Reduce (Identify unique faces and build edges)
    cdef vector[int] unique_faces_flat
    cdef vector[float] unique_faces_weights # S_faces
    
    # Pre-allocate Edges Vectors (Exact size known: N * K)
    cdef vector[int] edges_u
    cdef vector[int] edges_v
    cdef vector[float] edges_w
    cdef Py_ssize_t n_total_edges = n_simplexes * K
    edges_u.resize(n_total_edges)
    edges_v.resize(n_total_edges)
    edges_w.resize(n_total_edges)
    
    # Buffer to store FaceID for each (Simplex, Drop) to build edges later
    # Mapped by: mapped_face_ids[simplex_idx * (K+1) + drop_idx]
    cdef vector[int] mapped_face_ids
    mapped_face_ids.resize(total_faces)
    
    cdef Py_ssize_t current_idx = 0
    cdef Py_ssize_t current_group_start = 0
    cdef int face_id = 0
    cdef float w_sum, w_simp, inv_w_safe
    cdef int s_idx, d_idx, k_idx, p_idx
    cdef int u_val, v_val
    cdef float w_val
    cdef int base_off_edge
    
    with nogil:
        while current_idx < total_faces:
            current_group_start = current_idx
            w_sum = 0.0
            
            # Find group of identical faces
            while True:
                # Accumulate weight
                s_idx = all_faces[current_idx].simplex_idx
                w_simp = simplex_weights[s_idx]
                
                if w_simp > 1e-12:
                    inv_w_safe = 1.0 / w_simp
                else:
                    inv_w_safe = 1e12
                w_sum += inv_w_safe
                
                current_idx += 1
                if current_idx >= total_faces:
                    break
                
                # Check if next face is different
                if comp(all_faces[current_group_start], all_faces[current_idx]):
                    break
            
            # End of Group
            
            # 1. Store Unique Face Data (Vertices from first ref)
            s_idx = all_faces[current_group_start].simplex_idx
            d_idx = all_faces[current_group_start].drop_idx
            
            # Extract K vertices (skipping d_idx)
            for k_idx in range(K_plus_1):
                if k_idx == d_idx: continue
                unique_faces_flat.push_back(simplex_indices[s_idx, k_idx])
            
            # 2. Store Weight
            unique_faces_weights.push_back(w_sum)
            
            # 3. Map Face ID back to (Simplex, Drop)
            for k_idx in range(current_group_start, current_idx):
                s_idx = all_faces[k_idx].simplex_idx
                d_idx = all_faces[k_idx].drop_idx
                mapped_face_ids[s_idx * K_plus_1 + d_idx] = face_id
                
            face_id += 1

        # 4. Build Edges (Simplex-wise) - Parallel
        # Connect faces 0-1, 1-2, ... for each simplex
        for i in prange(n_simplexes, schedule='static'):
            w_val = simplex_weights[i]
            # Offset in mapped_face_ids
            idx = i * K_plus_1
            # Offset in edges arrays
            base_off_edge = i * K
            
            for j in range(K):
                u_val = mapped_face_ids[idx + j]
                v_val = mapped_face_ids[idx + j + 1]
                edges_u[base_off_edge + j] = u_val
                edges_v[base_off_edge + j] = v_val
                edges_w[base_off_edge + j] = w_val
                
    # --- Convert to Numpy ---
    cdef Py_ssize_t n_unique = unique_faces_weights.size()
    cdef Py_ssize_t n_edges = edges_w.size()
    
    # 1. Faces Unique
    cdef np.ndarray[ITYPE_t, ndim=2] faces_unique_arr = np.zeros((n_unique, K), dtype=np.int32)
    cdef Py_ssize_t flat_ptr = 0
    # Copy manually or using memcpy (loop is safe)
    for i in range(n_unique):
        for j in range(K):
            faces_unique_arr[i, j] = unique_faces_flat[flat_ptr]
            flat_ptr += 1
            
    # 2. S_faces
    cdef np.ndarray[DTYPE_t, ndim=1] s_faces_arr = np.zeros(n_unique, dtype=np.float32)
    for i in range(n_unique):
        s_faces_arr[i] = unique_faces_weights[i]
        
    # 3. Edges
    cdef np.ndarray[ITYPE_t, ndim=1] e_u_arr = np.zeros(n_edges, dtype=np.int32)
    cdef np.ndarray[ITYPE_t, ndim=1] e_v_arr = np.zeros(n_edges, dtype=np.int32)
    cdef np.ndarray[DTYPE_t, ndim=1] e_w_arr = np.zeros(n_edges, dtype=np.float32)
    
    for i in range(n_edges):
        e_u_arr[i] = edges_u[i]
        e_v_arr[i] = edges_v[i]
        e_w_arr[i] = edges_w[i]
    
    return faces_unique_arr, e_u_arr, e_v_arr, e_w_arr, s_faces_arr, n_unique


cdef void combinations_k_plus_1_indices(
    Py_ssize_t n, 
    int K, 
    vector[vector[Py_ssize_t]]& out_indices
) nogil:
    """
    Generates all combinations of indices 0..n-1 of size K+1.
    """
    cdef int k = K + 1
    if k > n:
        return
        
    cdef vector[Py_ssize_t] v
    v.resize(k)
    
    cdef int i
    for i in range(k):
        v[i] = i
        
    cdef int j
    
    while True:
        out_indices.push_back(v)
        
        if v[k-1] < n - 1:
            v[k-1] += 1
        else:
            j = k - 2
            while j >= 0 and v[j] >= n - k + j:
                j -= 1
            if j < 0:
                break
            v[j] += 1
            for i in range(j+1, k):
                v[i] = v[i-1] + 1