"""
condense_tree on a MST (HDBSCAN-like condensed tree)
=====================================================

This file implements a condensed tree builder on top of a (Euclidean or generic) Minimum Spanning Tree.

Inputs
------
- W_nodes: numpy.ndarray, float shape (N,). Node weights (e.g., 1 per point). Used for eligibility and stability weighting.
- U_mst, V_mst: numpy.ndarray, int shape (M,), M = N-1. Endpoints of MST edges.
- W_mst: numpy.ndarray, float shape (M,), edge weights sorted ascending (non-decreasing!).
- min_cluster_size: int. Cluster eligibility threshold: sum(W_nodes) >= min_cluster_size.

Definitions and behavior
------------------------
We scan MST edges by increasing W_mst. Components are maintained with a union-find. A component is eligible when its
accumulated weight >= min_cluster_size. When a component becomes eligible, we instantiate a *cluster leaf* in Z.
When two eligible components merge at an edge of weight r, we create a *new cluster* in Z with the two children.
Multiple edges may share the same weight r; processing in any order is fine because all relevant lambdas equal 1/r.

Stability (HDBSCAN spirit)
--------------------------
Let lambda = 1 / (r + eps). Each node contributes to the stability of an eligible cluster from the time it *joins* that
cluster until the cluster's *death* (when it merges with another eligible cluster), i.e. contribution = lambda_join - lambda_death.
We avoid per-node bookkeeping by maintaining per-cluster:
- n[j]: current total node weight in the cluster (float, but typically integer when W_nodes are 1)
- S_join[j]: sum over current members of lambda_join values
When cluster j dies at lambda_death we add stability[j] += S_join[j] - n[j]*lambda_death, then all its members join the
new parent cluster at lambda_death (so S_join[parent] starts/accumulates with n[j]*lambda_death, etc.). If the root cluster
never dies, we finalize stability[root] += S_join[root] (lambda_death = 0).

Arêtes_éligibles
----------------
For each cluster j we store the list of MST edge indices that participated in forming j during its lifetime, including:
- all edges that merged its (eligible) children,
- all edges that came from non-eligible components that attached to j before its death.

Return structure
----------------
The function returns a dictionary Z with:
- 'children': list[list[int]]; children clusters that formed each cluster j (empty for leaves)
- 'r': np.ndarray (#clusters,), the radius/edge weight at which the cluster j was created (birth r)
- 'stability': np.ndarray (#clusters,), the final stability of cluster j
- 'edges': list[list[int]]; indices into the MST edge order accumulated for each cluster j (Arêtes_éligibles)
- 'size': np.ndarray (#clusters,), the total W_nodes weight in cluster j at birth
- 'lambda_birth': np.ndarray (#clusters,), 1/(r+eps)
- 'lambda_death': np.ndarray (#clusters,), lambda where the cluster died (0 if it survived to the end)
- plus convenience copies of the MST and sizes for downstream selection: 'U','V','W','N','M'.

Additionally, helpers are provided to:
- compute a Euclidean MST for 2D points without external deps (Prim O(N^2))
- extract labels by cutting the MST at a given threshold and filtering components by min_cluster_size
- convert the condensed tree into clusters with `GetClusters` (EOM/leaf/DBSCAN-like) and an optional recursive `splitting` loss

Plotting note
-------------
This file is safe to run in environments without matplotlib. If matplotlib is available, demo plots will be shown;
otherwise, text summaries are printed instead.
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Tuple, Any

EPS = 1e-12


# ======================
# Core: condensed tree
# ======================

def condense_tree(
    W_nodes: np.ndarray,
    U_mst: np.ndarray,
    V_mst: np.ndarray,
    W_mst: np.ndarray,
    min_cluster_size: int,
    check_sorted: bool = True,
) -> Dict[str, Any]:
    """Build a HDBSCAN-like condensed tree directly from a MST.

    Parameters
    ----------
    W_nodes : (N,) float
        Node weights.
    U_mst, V_mst : (M,) int
        Endpoints of MST edges.
    W_mst : (M,) float
        Edge weights, sorted in non-decreasing order (ascending).
    min_cluster_size : int
        Minimum sum of W_nodes required for a component to become an eligible cluster.
    check_sorted : bool
        If True, validates that W_mst is non-decreasing.

    Returns
    -------
    Z : dict
        Condensed tree structure as described in the module docstring.
    """
    W_nodes = np.asarray(W_nodes, dtype=float)
    U_mst = np.asarray(U_mst, dtype=np.int64)
    V_mst = np.asarray(V_mst, dtype=np.int64)
    W_mst = np.asarray(W_mst, dtype=float)

    N = W_nodes.shape[0]
    M = W_mst.shape[0]
    if not (U_mst.shape[0] == V_mst.shape[0] == M):
        raise ValueError("U_mst, V_mst, W_mst must have same length M")
    if N != M + 1:
        raise ValueError(f"Expected N = M + 1, got N={N}, M={M}")
    if check_sorted and np.any(W_mst[1:] < W_mst[:-1]):
        raise ValueError("W_mst must be sorted in non-decreasing order")

    # Union-Find (Disjoint Set Union) with path compression
    parent = np.arange(N, dtype=np.int64)
    comp_weight = W_nodes.copy()  # sum of W_nodes per component root

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # For ineligible components, accumulate the list of MST edge indices used so far
    comp_edges: List[List[int]] = [[] for _ in range(N)]

    # Map component root -> current eligible cluster id (or -1 if ineligible)
    comp_cid = -np.ones(N, dtype=np.int64)

    # Per-cluster data (indexed by cluster id j)
    children: List[List[int]] = []
    birth_r: List[float] = []
    death_r: List[float] = []  # None -> np.nan internally, will be mapped to 0 for lambda_death
    stability: List[float] = []
    cluster_edges: List[List[int]] = []
    size_at_birth: List[float] = []
    n_in_cluster: List[float] = []
    sum_join_lambda: List[float] = []

    def new_leaf_from_component(root: int, r_birth: float, lam: float):
        """Promote an ineligible component to an eligible *leaf* cluster at radius r_birth.
        All current members join at lambda=lam, contributing later until the cluster dies.
        """
        cid = len(children)
        children.append([])
        birth_r.append(float(r_birth))
        death_r.append(np.nan)
        stability.append(0.0)
        cluster_edges.append(list(comp_edges[root]))
        n = float(comp_weight[root])
        size_at_birth.append(n)
        n_in_cluster.append(n)
        sum_join_lambda.append(n * lam)
        comp_edges[root].clear()
        comp_cid[root] = cid
        return cid

    # Process edges in ascending order
    for i in range(M):
        u = int(U_mst[i])
        v = int(V_mst[i])
        r = float(W_mst[i])
        lam = 1.0 / (r + EPS)

        ru = find(u)
        rv = find(v)
        if ru == rv:
            continue

        elig_u = comp_cid[ru] != -1
        elig_v = comp_cid[rv] != -1

        if not elig_u and not elig_v:
            # Union by weight to keep trees shallow
            if comp_weight[ru] < comp_weight[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            comp_weight[ru] += comp_weight[rv]
            # Aggregate edge indices
            if comp_edges[rv]:
                comp_edges[ru].extend(comp_edges[rv])
                comp_edges[rv].clear()
            comp_edges[ru].append(i)
            # If just became eligible, create leaf cluster
            if comp_cid[ru] == -1 and comp_weight[ru] >= min_cluster_size:
                new_leaf_from_component(ru, r_birth=r, lam=lam)

        elif elig_u and not elig_v:
            # Attach ineligible rv to eligible ru
            parent[rv] = ru
            comp_weight[ru] += comp_weight[rv]
            # Update cluster edges and stability join sums
            cid = int(comp_cid[ru])
            if comp_edges[rv]:
                cluster_edges[cid].extend(comp_edges[rv])
                comp_edges[rv].clear()
            cluster_edges[cid].append(i)
            n_in = float(comp_weight[rv])
            n_in_cluster[cid] += n_in
            sum_join_lambda[cid] += n_in * lam

        elif not elig_u and elig_v:
            # Attach ineligible ru to eligible rv
            parent[ru] = rv
            comp_weight[rv] += comp_weight[ru]
            cid = int(comp_cid[rv])
            if comp_edges[ru]:
                cluster_edges[cid].extend(comp_edges[ru])
                comp_edges[ru].clear()
            cluster_edges[cid].append(i)
            n_in = float(comp_weight[ru])
            n_in_cluster[cid] += n_in
            sum_join_lambda[cid] += n_in * lam

        else:
            # Both eligible -> close children at this lambda and create a new parent cluster
            cid_u = int(comp_cid[ru])
            cid_v = int(comp_cid[rv])

            # Close children (death at r), accumulate stability
            for cid in (cid_u, cid_v):
                if np.isnan(death_r[cid]):
                    death_r[cid] = r
                    stability[cid] += (sum_join_lambda[cid] - n_in_cluster[cid] * lam)

            # Create parent cluster at r
            cid_new = len(children)
            children.append([cid_u, cid_v])
            birth_r.append(r)
            death_r.append(np.nan)
            stability.append(0.0)
            edges_new: List[int] = []
            if cluster_edges[cid_u]:
                edges_new.extend(cluster_edges[cid_u])
            if cluster_edges[cid_v]:
                edges_new.extend(cluster_edges[cid_v])
            edges_new.append(i)
            cluster_edges.append(edges_new)
            n_parent = float(n_in_cluster[cid_u] + n_in_cluster[cid_v])
            size_at_birth.append(n_parent)
            n_in_cluster.append(n_parent)
            sum_join_lambda.append(n_parent * lam)

            # Union components; choose new root by weight
            if comp_weight[ru] < comp_weight[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            comp_weight[ru] += comp_weight[rv]
            comp_cid[ru] = cid_new
            comp_cid[rv] = -1

    # Finalize stability for clusters that never died (death lambda = 0)
    lambda_birth = np.array([1.0 / (rb + EPS) for rb in birth_r], dtype=float)
    lambda_death = np.zeros(len(children), dtype=float)
    for j in range(len(children)):
        if np.isnan(death_r[j]):
            stability[j] += sum_join_lambda[j]
            lambda_death[j] = 0.0
        else:
            lambda_death[j] = 1.0 / (death_r[j] + EPS)

    Z = {
        'children': children,
        'r': np.asarray(birth_r, dtype=float),
        'stability': np.asarray(stability, dtype=float),
        'edges': cluster_edges,
        'size': np.asarray(size_at_birth, dtype=float),
        'lambda_birth': lambda_birth,
        'lambda_death': lambda_death,
        # Convenience for downstream selection/cuts
        'U': U_mst.copy(),
        'V': V_mst.copy(),
        'W': W_mst.copy(),
        'N': int(N),
        'M': int(M),
    }
    return Z


# =====================================
# Convert Z to clusters and selections
# =====================================

def _compute_nodes_all(Z: Dict[str, Any]) -> List[np.ndarray]:
    """Return per-cluster list of node indices (sorted unique). Reconstructs from Z['edges'] and MST (U,V).
    Assumes every cluster's edges stay internal to that cluster, which holds for this construction.
    """
    U = Z['U']; V = Z['V']; M = Z['M']
    nodes_all: List[np.ndarray] = []
    for edges in Z['edges']:
        if not edges:
            nodes_all.append(np.empty(0, dtype=np.int64))
            continue
        idx = np.asarray(edges, dtype=np.int64)
        idx = idx[(idx >= 0) & (idx < M)]
        pts = np.unique(np.r_[U[idx], V[idx]].astype(np.int64))
        nodes_all.append(pts)
    return nodes_all


def _roots_of_Z(Z: Dict[str, Any]) -> List[int]:
    children = Z['children']
    K = len(children)
    is_child = np.zeros(K, dtype=bool)
    for j, ch in enumerate(children):
        for c in ch:
            if 0 <= c < K:
                is_child[c] = True
    roots = [j for j in range(K) if not is_child[j]]
    return roots


def _eom_select(Z: Dict[str, Any]) -> List[int]:
    """Excess-of-Mass style selection (maximize sum of stabilities over disjoint clusters)."""
    children = Z['children']
    stab = Z['stability']
    from functools import lru_cache

    @lru_cache(maxsize=None)
    def best_under(j: int) -> Tuple[Tuple[int, ...], float]:
        ch = children[j]
        if not ch:
            return (j,), float(stab[j])
        sel_l, val_l = best_under(ch[0])
        sel_r, val_r = best_under(ch[1])
        if val_l + val_r > stab[j]:
            return tuple(sel_l + sel_r), float(val_l + val_r)
        else:
            return (j,), float(stab[j])

    selected: List[int] = []
    for r in _roots_of_Z(Z):
        s, _ = best_under(r)
        selected.extend(list(s))
    return sorted(set(selected))


def GetClusters(Z: Dict[str, Any], method, splitting=None, verbose: bool = False) -> Dict[str, Any]:
    """Return clusters as lists of point indices according to a selection method and optional recursive splitting.

    Parameters
    ----------
    Z : dict
        Output of condense_tree (must include 'U','V','W','N','M').
    method : {'eom','leaf', float>0}
        'eom' for stability-based selection; 'leaf' for all eligible leaves; float r_cut for DBSCAN-like cut on MST.
    splitting : callable or None
        Optional loss function f(nodes: np.ndarray)->float. If provided, recursively split a chosen cluster j into its
        children whenever sum(loss(children)) <= loss(parent). This recurses until no further split reduces loss.
    verbose : bool

    Returns
    -------
    dict with keys:
      - 'clusters': List[np.ndarray] of node indices
      - 'cids': List[Optional[int]] cluster ids in Z (None when no direct Z node matches)
      - 'method': echoed method
    """
    N = int(Z['N'])
    U = Z['U']; V = Z['V']; W = Z['W']
    children = Z['children']

    # Precompute node membership per Z node for potential splitting and mapping
    nodes_per_cid = _compute_nodes_all(Z)
    set2cid = {tuple(arr.tolist()): j for j, arr in enumerate(nodes_per_cid) if arr.size > 0}

    from functools import lru_cache

    @lru_cache(maxsize=None)
    def _apply_splitting_on_cid(cid: int) -> Tuple[Tuple[Tuple[int, ...], ...], float]:
        nodes = nodes_per_cid[cid]
        loss_here = float(splitting(nodes))
        ch = children[cid]
        if not ch:
            return (tuple(nodes.tolist()),), loss_here
        left_nodes, loss_left = _apply_splitting_on_cid(ch[0])
        right_nodes, loss_right = _apply_splitting_on_cid(ch[1])
        if loss_left + loss_right <= loss_here:
            return left_nodes + right_nodes, float(loss_left + loss_right)
        else:
            return (tuple(nodes.tolist()),), loss_here

    # Initial selection by method
    selected_cids: List[int] = []
    clusters_nodes: List[np.ndarray] = []
    clusters_cids: List[Any] = []

    if isinstance(method, str):
        if method == 'leaf':
            selected_cids = [j for j, ch in enumerate(children) if not ch]
        elif method == 'eom':
            selected_cids = _eom_select(Z)
        else:
            raise ValueError("method must be 'eom', 'leaf', or a positive float")
        if splitting is None:
            for cid in selected_cids:
                clusters_nodes.append(nodes_per_cid[cid])
                clusters_cids.append(cid)
        else:
            for cid in selected_cids:
                tuples_list, _ = _apply_splitting_on_cid(cid)
                for tnd in tuples_list:
                    nd = np.asarray(tnd, dtype=np.int64)
                    clusters_nodes.append(nd)
                    clusters_cids.append(set2cid.get(tnd, None))

    else:
        # Float threshold: cut MST at r_cut
        r_cut = float(method)
        # Build components by DSU
        parent = np.arange(N, dtype=np.int64)
        size = np.ones(N, dtype=np.int64)

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        def union(a: int, b: int):
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]

        for i in range(len(W)):
            if W[i] <= r_cut:
                union(int(U[i]), int(V[i]))
            else:
                break
        roots = np.array([find(i) for i in range(N)], dtype=np.int64)
        for r in np.unique(roots):
            nodes = np.sort(np.where(roots == r)[0].astype(np.int64))
            clusters_nodes.append(nodes)
            clusters_cids.append(set2cid.get(tuple(nodes.tolist()), None))
        if splitting is not None:
            final_nodes: List[np.ndarray] = []
            final_cids: List[Any] = []
            for nd, cid in zip(clusters_nodes, clusters_cids):
                if cid is None:
                    final_nodes.append(nd)
                    final_cids.append(None)
                else:
                    tuples_list, _ = _apply_splitting_on_cid(cid)
                    for tnd in tuples_list:
                        nd2 = np.asarray(tnd, dtype=np.int64)
                        final_nodes.append(nd2)
                        final_cids.append(set2cid.get(tnd, None))
            clusters_nodes, clusters_cids = final_nodes, final_cids

    if verbose:
        print(f"[GetClusters] method={method} -> {len(clusters_nodes)} clusters")

    return {'clusters': clusters_nodes, 'cids': clusters_cids, 'method': method}
    # labels = -np.ones(N, dtype=int)
    # for idx, nd in enumerate(res['clusters']):
    #     labels[np.asarray(nd, dtype=int)] = idx

