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

# ======================
# Core: condensed tree
# ======================

from ._cython import condense_tree_cython, build_leaf_dfs_intervals

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
    epsilon: float = 0.0,
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
    epsilon : float
        Tolerance for merging edges with similar weights into a single N-ary node.

    Returns
    -------
    Z : dict
        Condensed tree structure as described in the module docstring.
    """
    # Delegate to the optimized Cython implementation
    return condense_tree_cython(
        W_nodes, U_mst, V_mst, W_mst, min_cluster_size, check_sorted, epsilon
    )

# The original pure Python implementation is preserved below for reference.
# It is commented out to ensure the optimized Cython version is used.
"""
def condense_tree(
    W_nodes: np.ndarray,
    U_mst: np.ndarray,
    V_mst: np.ndarray,
    W_mst: np.ndarray,
    min_cluster_size: int,
    check_sorted: bool = True,
) -> Dict[str, Any]:
    # ... (Original docstring) ...
    pass 
"""


# =====================================
# Convert Z to clusters and selections
# =====================================

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
    """Excess-of-Mass style selection (maximize sum of stabilities over disjoint clusters).
    Iterative implementation to avoid RecursionError on deep trees.
    """
    children = Z['children']
    stab = Z['stability']
    n_clusters = len(children)
    
    # We need to compute 'val' (max stability of subtree) and 'selection' for each node.
    # Since n_clusters can be large, we use arrays/lists.
    # selection[i] will store a list of selected cluster indices for the subtree rooted at i.
    # However, storing full lists for every node is O(N^2) memory in worst case (a line).
    # Optimization: We only need the list if we select the children. If we select the node itself, the list is just [i].
    # Actually, we only need to know: "Do we select this node i?" (bool)
    # If yes, we discard children selections. If no, we keep children selections.
    # So we can just compute a boolean array `is_selected`.
    # But wait, EOM is global.
    # Let V[i] be the max stability sum for the subtree at i.
    # V[i] = max(stab[i], V[left] + V[right])
    # If stab[i] > sum(V[children]), we mark i as selected (and unmark descendants).
    
    # 1. Compute V[i] bottom-up (Post-order traversal)
    # We can use an iterative post-order or just iterate backwards if indices are topological?
    # In condense_tree, parents are appended after children. So iterating n_clusters-1 down to 0 ?
    # Let's verify: "cid_new = len(children)". Yes, parents always have higher index than children.
    # So reverse iteration is a valid topological sort.
    
    max_stab = np.array(stab, dtype=np.float64) # Initialize with self stability
    # We also need to track WHICH choice we made (Self vs Children)
    # let's use a boolean array: keep_self[i] = True if stab[i] >= V[children]
    keep_self = np.ones(n_clusters, dtype=bool)
    
    # Iterate forward (0 to n_clusters-1).
    # Since children indices < parent indices, this ensures Children are processed BEFORE Parents (Bottom-Up).
    for i in range(n_clusters):
        ch = children[i]
        if ch:
            # Sum of max_stabilities of children
            sum_children_stab = 0.0
            for c in ch:
                sum_children_stab += max_stab[c]
            
            if sum_children_stab > max_stab[i]:
                max_stab[i] = sum_children_stab
                keep_self[i] = False
            else:
                # max_stab[i] remains stab[i]
                keep_self[i] = True
        else:
            # Leaf: max_stab is already stab[i], keep_self is True
            pass

    # 2. Collect selected clusters top-down
    # Start from roots. If a node is kept, add it and stop. Else recurse to children.
    selected = []
    stack = list(_roots_of_Z(Z))
    
    while stack:
        curr = stack.pop()
        if keep_self[curr]:
            selected.append(curr)
        else:
            # Propagate to children
            ch = children[curr]
            if ch:
                stack.extend(ch)
            else:
                # Should not happen if keep_self is False (implies children existed and had better score)
                # But for safety:
                selected.append(curr)
                
    return sorted(selected)


def _build_dfs_structure(Z: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares the O(1) node extraction structure using Full Post-Order Traversal.
    """
    children = Z['children']
    initial_membership = Z.get('initial_membership')
    if initial_membership is None:
        raise ValueError("Z does not contain 'initial_membership'. Re-run condense_tree.")

    n_clusters = len(children)
    
    dfs_rank = np.full(n_clusters, -1, dtype=np.int64)
    min_dfs = np.full(n_clusters, -1, dtype=np.int64)
    max_dfs = np.full(n_clusters, -1, dtype=np.int64)
    
    # Iterative Post-Order
    visit_stack = list(_roots_of_Z(Z))
    process_stack = []
    
    while visit_stack:
        curr = visit_stack.pop()
        process_stack.append(curr)
        ch = children[curr]
        if ch:
            for c in ch:
                visit_stack.append(c)
    
    current_rank_cursor = 0
    
    # Iterate in Post-Order (Children -> Root)
    for curr in reversed(process_stack):
        ch = children[curr]
        
        my_rank = current_rank_cursor
        current_rank_cursor += 1
        dfs_rank[curr] = my_rank
        
        if not ch:
            # Leaf
            min_dfs[curr] = my_rank
            max_dfs[curr] = my_rank + 1
        else:
            child_starts = [min_dfs[c] for c in ch]
            min_dfs[curr] = min(child_starts)
            max_dfs[curr] = my_rank + 1

    # 2. Sort the actual points
    mask_noise = initial_membership == -1
    valid_membership = initial_membership[~mask_noise]
    
    # Valid membership must be within [0, n_clusters-1]
    # Check bounds (DEBUG)
    # if len(valid_membership) > 0:
    #    print(f"DEBUG: valid_membership range [{valid_membership.min()}, {valid_membership.max()}]")
        
    valid_ranks = dfs_rank[valid_membership]
    
    # Check validity of ranks (DEBUG)
    # n_invalid_ranks = (valid_ranks == -1).sum()
    # if n_invalid_ranks > 0:
    #    print(f"DEBUG: FOUND {n_invalid_ranks} points pointing to unvisited clusters!")
    
    full_ranks = np.full(len(initial_membership), -1, dtype=np.int64)
    full_ranks[~mask_noise] = valid_ranks
    
    order = np.argsort(full_ranks)
    sorted_ranks = full_ranks[order]
    
    start_valid_idx = np.searchsorted(sorted_ranks, 0)
    
    n_ranks = n_clusters # Rank can go up to n_clusters-1
    # Actually current_rank_cursor is the number of ranks used. It should equal len(process_stack).
    # If process_stack < n_clusters, some nodes are unreachable (forest?) -> they stay rank -1.
    
    if n_ranks > 0:
        counts = np.bincount(sorted_ranks[start_valid_idx:], minlength=n_ranks)
        
        offsets = np.zeros(n_ranks + 1, dtype=np.int64)
        offsets[0] = start_valid_idx
        np.cumsum(counts, out=offsets[1:])
        
        cluster_start = np.take(offsets, min_dfs)
        cluster_end = np.take(offsets, max_dfs)
        
        # Handle nodes that were not visited (rank -1)
        # min_dfs/max_dfs are -1 for them.
        # take(-1) takes the last element. We should handle this.
        # Set their range to empty [0,0] or similar.
        mask_unvisited = (min_dfs == -1)
        cluster_start[mask_unvisited] = 0
        cluster_end[mask_unvisited] = 0
        
        return order, cluster_start, cluster_end
        
    else:
        return np.arange(len(initial_membership)), np.zeros(n_clusters, dtype=int), np.zeros(n_clusters, dtype=int)


def GetClusters(Z: Dict[str, Any], method, splitting=None, points=None, Face_to_points=None, verbose: bool = False) -> Dict[str, Any]:
    """Return clusters as lists of point indices according to a selection method and optional recursive splitting.

    Parameters
    ----------
    Z : dict
        Output of condense_tree.
    method : {'eom','leaf', float>0}
        'eom' for stability-based selection; 'leaf' for all eligible leaves; float r_cut for DBSCAN-like cut on MST.
    splitting : callable or None
        Optional decision function f(parent_idx: np.ndarray, children_idx_list: list[np.ndarray]) -> bool.
        Return True to split (descend), False to keep parent.
    verbose : bool

    Returns
    -------
    dict with keys:
      - 'clusters': List[np.ndarray] of node indices
      - 'cids': List[Optional[int]] cluster ids in Z
      - 'method': echoed method
    """
    N = int(Z['N'])
    children = Z['children']
    
    # 1. Build Optimized DFS Structure (One-time cost, very fast)
    nodes_ordered, c_start, c_end = _build_dfs_structure(Z)

    def get_nodes(cid: int) -> np.ndarray:
        s, e = c_start[cid], c_end[cid]
        return nodes_ordered[s:e]

    selected_cids: List[int] = []

    # 2. Select Clusters
    if isinstance(method, str):
        if method == 'leaf':
            selected_cids = [j for j, ch in enumerate(children) if not ch]
        elif method == 'eom':
            selected_cids = _eom_select(Z)
        else:
            raise ValueError("method must be 'eom', 'leaf', or a positive float")

    elif isinstance(method, (float, int)) and not isinstance(method, bool):
        # DBSCAN-like Cut using the hierarchy
        r_cut = float(method)
        lambda_cut = 1.0 / (r_cut + EPS) if r_cut > 0 else 1e20
        
        n_clusters = len(children)
        parent_map = np.full(n_clusters, -1, dtype=np.int64)
        for p, ch in enumerate(children):
            for c in ch:
                parent_map[c] = p
                
        lb = Z['lambda_birth']
        
        for i in range(n_clusters):
            # Check if this cluster exists at r_cut
            if lb[i] >= lambda_cut:
                # It is formed. Is it a root at this level?
                p = parent_map[i]
                if p == -1 or lb[p] < lambda_cut:
                    selected_cids.append(i)
        
        if verbose:
             print(f"[GetClusters] method={method} (cut) -> {len(selected_cids)} clusters found in hierarchy.")
             
    else:
        raise ValueError("Invalid method parameter")

    # 3. Extract & Split
    clusters_nodes = []
    clusters_cids = []

    if splitting is None:
        for cid in selected_cids:
            clusters_nodes.append(get_nodes(cid))
            clusters_cids.append(cid)
    else:
        # Splitting logic reused efficiently
        if Face_to_points is None:
            raise ValueError("Splitting requires 'Face_to_points'.")
        
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def _get_points_idx(cid: int) -> tuple:
            # Helper to cache point indices extraction
            nodes = get_nodes(cid)
            # Face_to_points is a numpy array (N_faces, K)
            faces_selected = Face_to_points[nodes]
            # Flatten and unique
            unique_pts = np.unique(faces_selected)
            # Filter -1 noise padding
            if unique_pts.size > 0 and unique_pts[0] < 0:
                unique_pts = unique_pts[unique_pts >= 0]
            return tuple(unique_pts.tolist())

        def _recursive_decision(cid: int) -> List[np.ndarray]:
            ch = children[cid]
            if not ch:
                # Leaf: cannot split further
                return [get_nodes(cid)]
            
            # Prepare data for splitting function
            # Parent points
            parent_pts_idx = np.array(_get_points_idx(cid), dtype=np.int64)
            
            # Children points
            children_pts_idx_list = []
            for child in ch:
                c_pts_idx = np.array(_get_points_idx(child), dtype=np.int64)
                children_pts_idx_list.append(c_pts_idx)
            
            # Call user function
            should_split = splitting(parent_pts_idx, children_pts_idx_list)
            
            if should_split:
                result = []
                for child in ch:
                    result.extend(_recursive_decision(child))
                return result
            else:
                return [get_nodes(cid)]

        for cid in selected_cids:
            # We treat each selected cluster as a root for potential splitting
            final_nodes_list = _recursive_decision(cid)
            for nd in final_nodes_list:
                clusters_nodes.append(nd)
                clusters_cids.append(None) # ID lost after splitting

    if verbose:
        print(f"[GetClusters] method={method} -> {len(clusters_nodes)} clusters")

    return {'clusters': clusters_nodes, 'cids': clusters_cids, 'method': method}


