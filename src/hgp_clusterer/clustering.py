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
    W_nodes : (N,) float32
        Node weights.
    U_mst, V_mst : (M,) int32
        Endpoints of MST edges.
    W_mst : (M,) float32
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
    # Explicit cast to 32-bit types
    return condense_tree_cython(
        W_nodes.astype(np.float32), 
        U_mst.astype(np.int32), 
        V_mst.astype(np.int32), 
        W_mst.astype(np.float32), 
        min_cluster_size, 
        check_sorted, 
        epsilon
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
    
    max_stab = np.array(stab, dtype=np.float32) # Initialize with self stability (float32)
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
    
    dfs_rank = np.full(n_clusters, -1, dtype=np.int32)
    min_dfs = np.full(n_clusters, -1, dtype=np.int32)
    max_dfs = np.full(n_clusters, -1, dtype=np.int32)
    
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
    
    full_ranks = np.full(len(initial_membership), -1, dtype=np.int32)
    full_ranks[~mask_noise] = valid_ranks
    
    order = np.argsort(full_ranks)
    sorted_ranks = full_ranks[order]
    
    start_valid_idx = np.searchsorted(sorted_ranks, 0)
    
    n_ranks = n_clusters # Rank can go up to n_clusters-1
    # Actually current_rank_cursor is the number of ranks used. It should equal len(process_stack).
    # If process_stack < n_clusters, some nodes are unreachable (forest?) -> they stay rank -1.
    
    if n_ranks > 0:
        counts = np.bincount(sorted_ranks[start_valid_idx:], minlength=n_ranks)
        
        offsets = np.zeros(n_ranks + 1, dtype=np.int32)
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
        
        return order.astype(np.int32), cluster_start, cluster_end
        
    else:
        return np.arange(len(initial_membership), dtype=np.int32), np.zeros(n_clusters, dtype=np.int32), np.zeros(n_clusters, dtype=np.int32)


def GetClusters(Z: Dict[str, Any], method, splitting=None, points=None, Face_to_points=None, S_faces=None, verbose: bool = False, whole_tree: bool = False) -> Dict[str, Any]:
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
    whole_tree : bool
        If True, returns a property graph of the entire subtree rooted at the selected clusters,
        using highly optimized contiguous NumPy arrays.

    Returns
    -------
    dict with keys:
      - If whole_tree=False: 'clusters', 'cids', 'method'
      - If whole_tree=True: 'tree_nodes', 'tree_parents', 'all_points', 'node_to_points_start', 'node_to_points_end', ...
    """
    N = int(Z['N'])
    children = Z['children']
    
    # 1. Build Optimized DFS Structure (One-time cost, very fast)
    nodes_ordered, c_start, c_end = _build_dfs_structure(Z)

    def get_nodes(cid: int) -> np.ndarray:
        s, e = c_start[cid], c_end[cid]
        nodes = nodes_ordered[s:e]
        if isinstance(method, (float, int)) and not isinstance(method, bool) and 'join_r' in Z:
            nodes = nodes[Z['join_r'][nodes] <= float(method)]
        return nodes

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
        lambda_cut = 1.0 / (r_cut + 1e-12) if r_cut > 0 else 1e20
        
        n_clusters = len(children)
        parent_map = np.full(n_clusters, -1, dtype=np.int32)
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

    # -------------------------------------------------------------------------
    # NEW OPTIMIZED PIPELINE: Data-Oriented "whole_tree"
    # -------------------------------------------------------------------------
    if whole_tree:
        if Face_to_points is None:
            raise ValueError("whole_tree requires 'Face_to_points'.")
        
        initial_membership = Z.get('initial_membership')
        if initial_membership is None:
            raise ValueError("Z must contain 'initial_membership' for whole_tree.")
        
        N_nodes = len(children)
        if S_faces is None:
            S_faces = np.ones(Face_to_points.shape[0], dtype=np.float32)
            
        # --- OPTIMIZATION: Only process selected subtrees ---
        # We only care about nodes that are descendants of `selected_cids`.
        valid_nodes_set = set()
        # Iterative DFS to avoid RecursionError on deep trees
        stack = list(selected_cids)
        while stack:
            curr = stack.pop()
            if curr not in valid_nodes_set:
                valid_nodes_set.add(curr)
                stack.extend(children[curr])

        # --- Phase 1: Direct Points & Weights ---
        valid_mask = initial_membership != -1
        if isinstance(method, (float, int)) and not isinstance(method, bool) and 'join_r' in Z:
            valid_mask &= (Z['join_r'] <= float(method))
            
        # EXTRA OPTIMIZATION: Filter down strictly to the subtrees
        valid_nodes_arr = np.array(list(valid_nodes_set), dtype=np.int32)
        if len(valid_nodes_arr) > 0:
            # np.isin is extremely fast and drops millions of irrelevant faces before memory allocation
            valid_mask &= np.isin(initial_membership, valid_nodes_arr)
        else:
            valid_mask[:] = False

        valid_faces = Face_to_points[valid_mask]
        valid_labels = initial_membership[valid_mask]
        valid_weights = S_faces[valid_mask]
        
        n_vertices = valid_faces.shape[1]
        direct_points = valid_faces.flatten()
        direct_labels = np.repeat(valid_labels, n_vertices)
        direct_weights = np.repeat(valid_weights, n_vertices)
        
        # Filter invalid vertices
        v_mask = direct_points >= 0
        direct_points = direct_points[v_mask]
        direct_labels = direct_labels[v_mask]
        direct_weights = direct_weights[v_mask]
        
        # Group by label (cid)
        order = np.argsort(direct_labels)
        direct_labels = direct_labels[order]
        direct_points = direct_points[order]
        direct_weights = direct_weights[order]
        
        unique_labels, split_indices = np.unique(direct_labels, return_index=True)
        split_indices = np.append(split_indices, len(direct_labels))
        
            
        # Prepare arrays for nodes
        # We use python lists of numpy arrays because arrays have variable length.
        # This is still 100% vectorized per node.
        node_pts = [np.array([], dtype=np.int32) for _ in range(N_nodes)]
        node_wgs = [np.array([], dtype=np.float32) for _ in range(N_nodes)]
        
        # Assign direct points and compute local unique weights
        for i, lbl in enumerate(unique_labels):
            if lbl not in valid_nodes_set:
                continue
            start = split_indices[i]
            end = split_indices[i+1]
            pts = direct_points[start:end]
            wgs = direct_weights[start:end]
            
            # Aggregate direct duplicates
            u_pts, inv = np.unique(pts, return_inverse=True)
            u_wgs = np.bincount(inv, weights=wgs).astype(np.float32)
            node_pts[lbl] = u_pts
            node_wgs[lbl] = u_wgs
            
        # --- Phase 2: Bottom-Up Weights Propagation ---
        # We still iterate 0 to N_nodes to guarantee topological order (bottom-up),
        # but we ONLY compute for valid_nodes.
        for cid in range(N_nodes):
            if cid not in valid_nodes_set:
                continue
            ch = children[cid]
            if len(ch) > 0:
                pts_list = [node_pts[cid]] + [node_pts[c] for c in ch]
                wgs_list = [node_wgs[cid]] + [node_wgs[c] for c in ch]
                
                cat_pts = np.concatenate(pts_list)
                cat_wgs = np.concatenate(wgs_list)
                
                if len(cat_pts) > 0:
                    order_pts = np.argsort(cat_pts)
                    cat_pts = cat_pts[order_pts]
                    cat_wgs = cat_wgs[order_pts]
                    
                    u_pts, inv = np.unique(cat_pts, return_inverse=True)
                    u_wgs = np.bincount(inv, weights=cat_wgs).astype(np.float32)
                    
                    node_pts[cid] = u_pts
                    node_wgs[cid] = u_wgs

        # --- Phase 3: Top-Down Strict Partition (Argmax) ---
        exclusive_pts = [np.array([], dtype=np.int32) for _ in range(N_nodes)]
        
        tree_nodes_list = []
        tree_parents_list = []
        
        # Iterative distribution and topology traversal
        for root_cid in selected_cids:
            stack = [(root_cid, -1, node_pts[root_cid])]
            
            while stack:
                cid, parent_id, P_parent = stack.pop()
                
                tree_nodes_list.append(cid)
                tree_parents_list.append(parent_id)
                
                ch = children[cid]
                if len(ch) == 0:
                    exclusive_pts[cid] = P_parent
                    continue
                    
                if len(P_parent) == 0:
                    # Still need to traverse topology even if points are empty
                    for c in reversed(ch):
                        stack.append((c, cid, np.array([], dtype=np.int32)))
                    continue
                    
                n_ch = len(ch)
                W_mat = np.zeros((len(P_parent), n_ch), dtype=np.float32)
                
                for i, c in enumerate(ch):
                    P_c = node_pts[c]
                    W_c = node_wgs[c]
                    if len(P_c) > 0:
                        idx_in_parent = np.searchsorted(P_parent, P_c)
                        valid_mask = idx_in_parent < len(P_parent)
                        valid_mask[valid_mask] &= (P_parent[idx_in_parent[valid_mask]] == P_c[valid_mask])
                        W_mat[idx_in_parent[valid_mask], i] = W_c[valid_mask]                
                best_ch_idx = np.argmax(W_mat, axis=1)
                max_w = np.max(W_mat, axis=1)
                
                parent_excl_mask = max_w == 0
                exclusive_pts[cid] = P_parent[parent_excl_mask]
                
                # Reverse to maintain original left-to-right order in DFS
                for i in reversed(range(len(ch))):
                    c = ch[i]
                    ch_mask = (best_ch_idx == i) & (~parent_excl_mask)
                    stack.append((c, cid, P_parent[ch_mask]))
            
        tree_nodes = np.array(tree_nodes_list, dtype=np.int32)
        tree_parents_cids = np.array(tree_parents_list, dtype=np.int32)
        
        # --- Phase 4: Contiguous Memory Layout (DFS) ---
        total_pts = sum(len(exclusive_pts[cid]) for cid in tree_nodes)
        all_points = np.empty(total_pts, dtype=np.int32)
        
        node_start = np.zeros(N_nodes, dtype=np.int32)
        node_end = np.zeros(N_nodes, dtype=np.int32)
        node_excl_start = np.zeros(N_nodes, dtype=np.int32)
        node_excl_end = np.zeros(N_nodes, dtype=np.int32)
        
        # Iterative Post-Order DFS for layout
        current_offset = 0
        for root_cid in selected_cids:
            stack = [(root_cid, False)]
            while stack:
                cid, is_post = stack.pop()
                if not is_post:
                    # Pre-visit: Write exclusive points
                    excl = exclusive_pts[cid]
                    n_excl = len(excl)
                    node_excl_start[cid] = current_offset
                    if n_excl > 0:
                        all_points[current_offset : current_offset + n_excl] = excl
                    current_offset += n_excl
                    node_excl_end[cid] = current_offset
                    node_start[cid] = node_excl_start[cid]
                    
                    # Push post-visit marker
                    stack.append((cid, True))
                    # Push children
                    for c in reversed(children[cid]):
                        stack.append((c, False))
                else:
                    # Post-visit: update end offset
                    node_end[cid] = current_offset
            
        # Map back to compact arrays matching `tree_nodes`
        node_to_idx = {cid: i for i, cid in enumerate(tree_nodes)}
        tree_parents_local = np.array([node_to_idx[p] if p != -1 else -1 for p in tree_parents_cids], dtype=np.int32)
        
        tree_node_start = np.array([node_start[cid] for cid in tree_nodes], dtype=np.int32)
        tree_node_end = np.array([node_end[cid] for cid in tree_nodes], dtype=np.int32)
        tree_node_excl_start = np.array([node_excl_start[cid] for cid in tree_nodes], dtype=np.int32)
        tree_node_excl_end = np.array([node_excl_end[cid] for cid in tree_nodes], dtype=np.int32)

        if verbose:
            print(f"[GetClusters] whole_tree=True -> {len(tree_nodes)} total tree nodes, {len(all_points)} total points.")
            
        return {
            "tree_nodes": tree_nodes,
            "tree_parents": tree_parents_local, 
            "tree_parents_cids": tree_parents_cids,
            "all_points": all_points,
            "node_to_points_start": tree_node_start,
            "node_to_points_end": tree_node_end,
            "node_exclusive_start": tree_node_excl_start,
            "node_exclusive_end": tree_node_excl_end,
            "method": method
        }

    # -------------------------------------------------------------------------
    # ORIGINAL PIPELINE (whole_tree=False)
    # -------------------------------------------------------------------------
    # 3. Extract & Split
    clusters_nodes = []
    clusters_cids = []

    if splitting is None:
        for cid in selected_cids:
            nodes = get_nodes(cid)
            if len(nodes) > 0:
                clusters_nodes.append(nodes)
                clusters_cids.append(cid)
    else:
        # Splitting logic reused efficiently
        if Face_to_points is None:
            raise ValueError("Splitting requires 'Face_to_points'.")
        
        initial_membership = Z.get('initial_membership')
        if initial_membership is None:
            raise ValueError("Z must contain 'initial_membership' for splitting optimization.")
            
        from functools import lru_cache
        
        if S_faces is None:
            S_faces = np.ones(Face_to_points.shape[0], dtype=np.float32)
            
        valid_mask = initial_membership != -1
        if isinstance(method, (float, int)) and not isinstance(method, bool) and 'join_r' in Z:
            valid_mask &= (Z['join_r'] <= float(method))

        valid_faces = Face_to_points[valid_mask]
        valid_labels = initial_membership[valid_mask]
        valid_S_faces = S_faces[valid_mask]
        
        order = np.argsort(valid_labels)
        sorted_faces = valid_faces[order]
        sorted_labels = valid_labels[order]
        sorted_S_faces = valid_S_faces[order]
        
        unique_labels, split_indices = np.unique(sorted_labels, return_index=True)
        
        leaf_states_map = {}
        n_groups = len(unique_labels)
        n_vertices_per_face = Face_to_points.shape[1]
        
        for i in range(n_groups):
            lbl = unique_labels[i]
            start = split_indices[i]
            end = split_indices[i+1] if i + 1 < n_groups else len(sorted_labels)
            
            faces_in_leaf = sorted_faces[start:end]
            S_faces_in_leaf = sorted_S_faces[start:end]
            
            points_concat = faces_in_leaf.flatten()
            weights_concat = np.repeat(S_faces_in_leaf, n_vertices_per_face)
            
            valid_pts_mask = points_concat >= 0
            if not valid_pts_mask.all():
                 points_concat = points_concat[valid_pts_mask]
                 weights_concat = weights_concat[valid_pts_mask]
                 
            if len(points_concat) > 0:
                P_leaf, inverse = np.unique(points_concat, return_inverse=True)
                W_leaf = np.bincount(inverse, weights=weights_concat).astype(np.float32)
                leaf_states_map[lbl] = (P_leaf.astype(np.int32), W_leaf)

        @lru_cache(maxsize=None)
        def _get_points_and_weights(cid: int):
            ch = children[cid]
            if not ch:
                return leaf_states_map.get(cid, (np.array([], dtype=np.int32), np.array([], dtype=np.float32)))
            
            child_states = [_get_points_and_weights(c) for c in ch]
            child_states = [s for s in child_states if len(s[0]) > 0]
            if not child_states:
                return np.array([], dtype=np.int32), np.array([], dtype=np.float32)
                
            P_all = np.concatenate([s[0] for s in child_states])
            W_all = np.concatenate([s[1] for s in child_states])
            
            P_parent, inverse = np.unique(P_all, return_inverse=True)
            W_parent = np.bincount(inverse, weights=W_all).astype(np.float32)
            
            return P_parent, W_parent

        def _recursive_decision(cid: int) -> List[np.ndarray]:
            ch = children[cid]
            if not ch:
                return [get_nodes(cid)]
            
            parent_pts, parent_weights = _get_points_and_weights(cid)
            
            if len(parent_pts) == 0:
                return [get_nodes(cid)]
            
            children_states = [_get_points_and_weights(child) for child in ch]
            
            best_child_idx = np.zeros(len(parent_pts), dtype=int)
            max_weights = np.full(len(parent_pts), -1.0, dtype=np.float32)
            
            for i, (P_c, W_c) in enumerate(children_states):
                if len(P_c) == 0:
                    continue
                idx_in_parent = np.searchsorted(parent_pts, P_c)
                valid_idx_mask = (idx_in_parent < len(parent_pts)) & (parent_pts[idx_in_parent] == P_c)
                
                idx_in_parent = idx_in_parent[valid_idx_mask]
                W_c_valid = W_c[valid_idx_mask]
                
                mask_better = W_c_valid > max_weights[idx_in_parent]
                
                best_child_idx[idx_in_parent[mask_better]] = i
                max_weights[idx_in_parent[mask_better]] = W_c_valid[mask_better]

            children_pts_list_disjoints = [
                parent_pts[best_child_idx == i] for i in range(len(ch))
            ]
            
            should_split = splitting(parent_pts, children_pts_list_disjoints)
            
            if should_split:
                result = []
                for child in ch:
                    result.extend(_recursive_decision(child))
                return result
            else:
                return [get_nodes(cid)]

        for cid in selected_cids:
            final_nodes_list = _recursive_decision(cid)
            for nd in final_nodes_list:
                clusters_nodes.append(nd)
                clusters_cids.append(None) 

    if verbose:
        print(f"[GetClusters] method={method} -> {len(clusters_nodes)} clusters")

    return {'clusters': clusters_nodes, 'cids': clusters_cids, 'method': method}
