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
    # Delegate to the optimized Cython implementation
    return condense_tree_cython(
        W_nodes, U_mst, V_mst, W_mst, min_cluster_size, check_sorted
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
    
    # Iterate backwards (leaves are already processed implicitly as they have no children)
    for i in range(n_clusters - 1, -1, -1):
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
    Prepares the O(1) node extraction structure.
    Returns:
        nodes_ordered: Array of node indices sorted by their leaf's DFS rank.
        first: Array of shape (n_clusters,) - start index in nodes_ordered for each cluster.
        last: Array of shape (n_clusters,) - end index (exclusive) in nodes_ordered for each cluster.
    """
    children = Z['children']
    initial_membership = Z.get('initial_membership')
    if initial_membership is None:
        raise ValueError("Z does not contain 'initial_membership'. Re-run condense_tree.")

    n_clusters = len(children)
    # 1. Build binary tree arrays for Cython
    # In condense_tree, new clusters are strictly binary merges (except leaves which have no children)
    # Indices are topological: children < parent.
    # However, to be safe with Cython logic which expects [0..t-1] leaves and [t..m+t-2] internal,
    # we need to be careful. Our Z indices are just 0..K-1 mixed.
    # But wait, our condense_tree logic (standard HDBSCAN) creates a structure where:
    # - Leaves are created first? Not necessarily.
    # - But parents are always created AFTER children.
    # The Cython `build_leaf_dfs_intervals` expects a specific SciPy-like linkage format:
    # leaves are 0..N-1, internal nodes N..2N-2.
    # OUR Z STRUCTURE IS NOT SCIPY-LIKE. It's a general list where children can be anywhere < current.
    #
    # ADAPTATION: We will map our Z indices to a SciPy-like structure for the traversal.
    # Or, simpler: We rewrite the DFS in Python here because the structure conversion cost
    # might outweigh the traversal gain for "only" millions of points but "only" thousands of clusters.
    # Actually, the number of clusters is O(N) worst case but usually much smaller.
    # Let's do a fast iterative DFS in Python/NumPy. It's safe for thousands of clusters.
    
    # Identify leaves in Z
    is_leaf = np.array([not bool(ch) for ch in children])
    leaves_z_idx = np.where(is_leaf)[0]
    n_leaves = len(leaves_z_idx)
    
    # Map Z_index -> Leaf_Rank (0..n_leaves-1) for leaves, -1 otherwise
    leaf_rank = np.full(n_clusters, -1, dtype=np.int64)
    leaf_rank[leaves_z_idx] = np.arange(n_leaves)
    
    # Compute interval [min_rank, max_rank] for each cluster
    # Iterative post-order traversal (bottom-up) is easy because parents > children in Z construction?
    # Actually condense_tree appends parents after children. So we can just iterate backwards or forwards?
    # Forward iteration 0..K-1 works perfectly for bottom-up accumulation!
    
    min_rank = np.full(n_clusters, 2**60, dtype=np.int64)
    max_rank = np.full(n_clusters, -1, dtype=np.int64)
    
    # Initialize leaves
    min_rank[leaves_z_idx] = leaf_rank[leaves_z_idx]
    max_rank[leaves_z_idx] = leaf_rank[leaves_z_idx] + 1 # Exclusive end
    
    # Propagate up
    # However, we need the specific DFS order for this to be a single interval.
    # Simply propagating min/max is only valid if the tree is planar/ordered such that leaves are contiguous.
    # We must enforce an ordering. A simple DFS traversal defines this ordering.
    
    # DFS to assign ranks
    # Recursive is risky. Iterative DFS:
    stack = list(_roots_of_Z(Z))
    # We need to visit leaves in order.
    # Pre-order traversal to collect leaves?
    # Actually we want the mapping: Leaf_Z_ID -> DFS_Order_Index
    
    dfs_order_leaves = []
    
    # Standard Iterative DFS
    visited = set()
    stack = list(reversed(_roots_of_Z(Z))) # Process roots
    while stack:
        curr = stack.pop()
        ch = children[curr]
        if not ch:
            dfs_order_leaves.append(curr)
        else:
            # Push children right then left so left is processed first
            stack.append(ch[1])
            stack.append(ch[0])
            
    # Now we have the correct permutation of leaves
    leaf_z_to_dfs_rank = np.full(n_clusters, -1, dtype=np.int64)
    dfs_order_leaves_arr = np.array(dfs_order_leaves, dtype=np.int64)
    leaf_z_to_dfs_rank[dfs_order_leaves_arr] = np.arange(n_leaves)
    
    # Now propagate intervals bottom-up
    # Since Z is topologically sorted (parents > children), we can just loop 0..K-1?
    # NO. Z indices are chronological by edge weight, not strictly topological in hierarchy depth?
    # Actually condense_tree appends new clusters. So a parent ALWAYS has a higher index than its children.
    # Proof: Parent created at merge of u, v. u and v must exist (have indices) before merge.
    # So iterating 0..K-1 is valid for leaves, but iterating 0..K-1 is also valid for bottom-up!
    
    min_dfs = np.full(n_clusters, 2**60, dtype=np.int64)
    max_dfs = np.full(n_clusters, -1, dtype=np.int64)
    
    # Initialize leaves with their new rank
    # Note: loop over all clusters to handle leaves and propagate
    for i in range(n_clusters):
        if not children[i]:
            rank = leaf_z_to_dfs_rank[i]
            if rank != -1: # Should be true
                min_dfs[i] = rank
                max_dfs[i] = rank + 1
        else:
            # Merge children intervals
            # Since we defined DFS order by traversing left then right child,
            # the interval is [min(left), max(right)]
            c1, c2 = children[i]
            # Because i > c1 and i > c2, c1 and c2 are already processed
            low = min(min_dfs[c1], min_dfs[c2])
            high = max(max_dfs[c1], max_dfs[c2])
            min_dfs[i] = low
            max_dfs[i] = high

    # 2. Sort the actual points
    # initial_membership contains Leaf_Z_IDs.
    # We map them to DFS_ranks.
    # Points with membership -1 (noise) get rank -1
    
    # Vectorized map
    membership_dfs_rank = np.full_like(initial_membership, -1)
    mask = initial_membership != -1
    # Only map valid memberships
    valid_members = initial_membership[mask]
    membership_dfs_rank[mask] = leaf_z_to_dfs_rank[valid_members]
    
    # Sort points by this rank
    # We want noise (-1) at the end or beginning?
    # argsort puts -1 at the beginning.
    # We need to know where the first real leaf starts.
    
    order = np.argsort(membership_dfs_rank)
    sorted_ranks = membership_dfs_rank[order]
    
    # Find start of non-noise
    # searchsorted finds the first index >= 0
    start_valid_idx = np.searchsorted(sorted_ranks, 0)
    
    # The leaves cover the range [0, n_leaves) in rank space.
    # In node array space, rank R starts at 'start_of_R' and ends at 'end_of_R'.
    # We can precompute these offsets efficiently.
    
    # Counts per rank
    # Bincount works on non-negative integers.
    if n_leaves > 0:
        counts = np.bincount(sorted_ranks[start_valid_idx:])
        # Check if counts matches n_leaves
        if len(counts) < n_leaves:
            # Pad if some leaves are empty (possible?)
            counts = np.pad(counts, (0, n_leaves - len(counts)))
        
        offsets = np.zeros(n_leaves + 1, dtype=np.int64)
        offsets[0] = start_valid_idx
        np.cumsum(counts, out=offsets[1:])
        
        # Now, for any cluster C, its points are nodes_ordered[offsets[min_dfs[C]] : offsets[max_dfs[C]]]
        # Because the interval [min_dfs, max_dfs) in ranks corresponds to range of offsets.
        
        # Map back to cluster arrays
        # We need efficient lookup.
        # Let's return the simplified arrays
        
        # Global arrays for GetClusters
        # Access: nodes_ordered[ cluster_start[cid] : cluster_end[cid] ]
        
        cluster_start = np.take(offsets, min_dfs) # shape (n_clusters,)
        cluster_end = np.take(offsets, max_dfs)   # shape (n_clusters,)
        
        # Safety for clusters that are effectively empty or invalid?
        # If min_dfs > max_dfs (should not happen), slice is empty.
        
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
        Optional loss function f(nodes: np.ndarray)->float.
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
        # A cluster is selected if it is "active" at r_cut.
        # Active means: born before r_cut AND (dies after r_cut OR never dies)
        # AND we usually want the most specific (deepest) clusters satisfying this,
        # but in a condensed tree, branches are disjoint.
        # If a parent is active, should we return it?
        # In HDBSCAN cut: we take the connected components of the graph at threshold.
        # In the condensed tree, this corresponds to the highest nodes (closest to root)
        # that are born <= r_cut.
        # Why? Because if a node is born <= r_cut, its children merged at that radius.
        # If it dies > r_cut, it persists.
        
        r_cut = float(method)
        lambda_cut = 1.0 / (r_cut + EPS) if r_cut > 0 else 1e20
        
        # We traverse top-down or check all nodes?
        # A node i is a root of a component at r_cut if:
        # 1. lambda_birth[i] >= lambda_cut (born at sufficiently small radius)
        # 2. Parent is NOT eligible (parent born at smaller lambda / larger radius).
        # Actually in condense_tree, lambda_birth is when the cluster FORMS (merges children).
        # So if lambda_birth[i] >= lambda_cut, the merge happened "tightly" enough.
        
        # Let's scan all nodes.
        # We need parent info to check if we are maximal.
        # Since parents > children, we can build parent map quickly.
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
                # It is a root if it has no parent OR parent is formed at lower lambda (larger radius)
                # i.e., parent merge hasn't happened yet at r_cut.
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
        if points is None or Face_to_points is None:
            raise ValueError("Splitting requires 'points' and 'Face_to_points'.")
        points = np.asarray(points)

        def _points_for_nodes(nodes: np.ndarray) -> np.ndarray:
            point_indices: set[int] = set()
            for node in nodes:
                pts = Face_to_points[int(node)]
                if pts is None: continue
                # Handle single or iterable
                if isinstance(pts, (np.ndarray, list, tuple, set)):
                    iterable = pts
                else:
                    iterable = (pts,)
                for p in iterable:
                    point_indices.add(int(p))
            if not point_indices:
                return points[[]]
            ordered_idx = np.fromiter(sorted(point_indices), dtype=np.int64)
            return points[ordered_idx]

        from functools import lru_cache

        @lru_cache(maxsize=None)
        def _apply_splitting_on_cid(cid: int) -> Tuple[Tuple[Tuple[int, ...], ...], float]:
            nodes = get_nodes(cid)
            points_cl = _points_for_nodes(nodes)
            loss_here = float(splitting(points_cl))
            
            ch = children[cid]
            if not ch:
                return (tuple(nodes.tolist()),), loss_here
            
            # Recursive check
            left_tuples, loss_left = _apply_splitting_on_cid(ch[0])
            right_tuples, loss_right = _apply_splitting_on_cid(ch[1])
            
            if loss_left + loss_right <= loss_here + EPS:
                return left_tuples + right_tuples, float(loss_left + loss_right)
            else:
                return (tuple(nodes.tolist()),), loss_here

        for cid in selected_cids:
            tuples_list, _ = _apply_splitting_on_cid(cid)
            for tnd in tuples_list:
                nd = np.asarray(tnd, dtype=np.int64)
                clusters_nodes.append(nd)
                clusters_cids.append(None)

    if verbose:
        print(f"[GetClusters] method={method} -> {len(clusters_nodes)} clusters")

    return {'clusters': clusters_nodes, 'cids': clusters_cids, 'method': method}


