from .clustering import GetClusters, condense_tree
from .hypergraph import _build_graph_KSimplexes
from .union_find import UnionFind
from ._cython import kruskal
from .geometry import propagate_labels_knn

import math
import numpy as np
import os

from sklearn.decomposition import PCA

from joblib import Parallel, delayed, cpu_count


def HypergraphPercol(
    M: np.ndarray,
    K: int = 2,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    metric: str = "euclidean",
    method = 'eom',
    splitting = None,
    weight_face: str = "lambda", # "lambda" ∝ 1/r ; "uniform" ∝ 1 ; "unique" 1 on the face with min r
    label_all_points: bool = False,
    return_multi_clusters: bool = False,
    complex_chosen: str = "auto",
    expZ: float = 2,
    precision: str = "safe",
    dim_reducer: bool | str = False,
    threshold_variance_dim_reduction: float = 0.999,
    verbeux: bool = False,
    cgal_root: str | os.PathLike[str] | None = "/content/HGP-clusterer/CGALDelaunay",
    subsample: float = 1.0,
) -> np.ndarray | tuple[np.ndarray, list[list[tuple[int, float, float]]]]:
    if method is None:
        method = "eom"
    is_sparse_metric = metric == "sparse"
    
    # 1. Handling Input Data & Subsampling
    if is_sparse_metric:
        M = np.asarray(M, dtype=np.float64)
        if M.ndim != 2 or M.shape[1] != 3:
            raise ValueError("For metric='sparse', M must be provided as (i, j, distance) triplets.")
        if M.size:
            n_full = int(np.max(M[:, :2])) + 1
        else:
            n_full = 0
        d_full = 0
        # Subsampling for sparse graph is tricky (node subsampling vs edge subsampling). 
        # For now, we assume subsample=1.0 or we ignore it/warn.
        if subsample < 1.0 and verbeux:
            print("Warning: Subsampling is not supported for sparse metric yet. Ignored.")
        subsample = 1.0
        X_core = M
        idx_core = np.arange(n_full)
    else:
        M_full = np.ascontiguousarray(M, dtype=np.float64)
        n_full, d_full = M_full.shape
        
        if 0.0 < subsample < 1.0 and n_full > 100: # Don't subsample tiny datasets
            n_core = int(n_full * subsample)
            if verbeux:
                print(f"Subsampling: selecting {n_core} points out of {n_full} ({subsample*100:.1f}%)")
            rng = np.random.default_rng(42) # Fixed seed for reproducibility or None? Let's fix it for stability.
            idx_core = rng.choice(n_full, size=n_core, replace=False)
            X_core = M_full[idx_core]
        else:
            idx_core = np.arange(n_full)
            X_core = M_full
    
    X = np.copy(X_core)
    n, d = (X.shape if not is_sparse_metric else (n_full, 0))
    
    if min_cluster_size is None:
        min_cluster_size = round(math.sqrt(n))
    
    pre = metric == "precomputed"
    delaunay_possible = not pre and metric == "euclidean" and not is_sparse_metric and M.ndim == 2
    if min_samples is None or min_samples <= K:
        min_samples = K + 1
    if n > 0:
        min_samples = min(min_samples, n)
        
    # 2. Dimensionality Reduction (on core set)
    if not is_sparse_metric and str(dim_reducer).lower() in {"pca", "umap"} and delaunay_possible:
        pca = PCA(n_components=threshold_variance_dim_reduction, svd_solver="full", whiten=False)
        X2 = pca.fit_transform(X)
        r = pca.n_components_
        ratio = pca.explained_variance_ratio_.sum()
        if r < d and str(dim_reducer).lower() == "pca":
            X = X2
            if verbeux:
                print(f"Dimension réduite par PCA : {d} → {r} (variance {ratio:.3f})")
        elif r < d and str(dim_reducer).lower() == "umap":
            from umap import UMAP

            reducer = UMAP(n_components=r, n_neighbors=max(2 * 2 * (K + 1), min_samples), metric=metric)
            X = reducer.fit_transform(X)
            if verbeux:
                print(f"Dimension réduite par UMAP : {d} → {r}")
    
    # 3. Build Hypergraph & Clustering
    faces_raw, e_u, e_v, e_w, faces_Simplexes, nS = _build_graph_KSimplexes(
        X,
        K,
        min_samples,
        metric,
        complex_chosen,
        expZ,
        precision=precision,
        verbose=verbeux,
        cgal_root=cgal_root,
    )
    if verbeux:
        print(f"{K}-simplices={nS}")
    if not faces_raw:
        if verbeux and not is_sparse_metric and K > d:
            print(
                "Warning: K too high compared to the dimension of the data. "
                "No clustering possible with such a K."
            )
        empty_labels = np.full(n_full, -1, dtype=np.int64)
        if return_multi_clusters:
            empty_multi = [[(-1, 1.0)] for _ in range(n_full)]
            return empty_labels, empty_multi
        return empty_labels
    faces_raw_arr = np.asarray(faces_raw, dtype=np.int64, order="C")
    e_u = np.asarray(e_u, dtype=np.int64)
    e_v = np.asarray(e_v, dtype=np.int64)
    e_w = np.asarray(e_w, dtype=np.float64)
    faces_unique, inv = np.unique(faces_raw_arr, axis=0, return_inverse=True)
    N = faces_unique.shape[0]
    if verbeux :
        print(
            f"Faces uniques: {N} (compression {faces_raw_arr.shape[0]}→{faces_unique.shape[0]})"
        )
    
    ### Ici répartir les poids des points sur les faces = (K-1)-simplexes
    Points = [[] for _ in range(n)]
    Face_to_points = [set() for _ in range(N)]
    for (old_idx, points_face, r_face) in faces_Simplexes :
        idx_face = inv[old_idx]
        for p in points_face :
            Face_to_points[idx_face].add(p)
            Points[p].append((idx_face,r_face))
    Points_w = [{} if weight_face == "uniform" or weight_face == "lambda" else [(-1, 0)] for _ in range(n)]
    for p,liste_faces in enumerate(Points) :
        for (idx_face,w_face) in liste_faces :
            if weight_face == "uniform" or weight_face == "lambda" :
                ajout = 1 if weight_face == "uniform" else 1/w_face
                if idx_face in Points_w[p] :
                    Points_w[p][idx_face] += ajout
                else :
                    Points_w[p][idx_face] = ajout
            elif weight_face == "unique" :
                if Points_w[p][0][1] < 1/w_face :
                    Points_w[p][0] = (idx_face, 1/w_face)
            else :
                1/0
        if weight_face == "uniform" or weight_face == "lambda" :
            liste_faces_w = list(Points_w[p].items())
            Points_w[p] = liste_faces_w
        somme = 0
        for _,s in Points_w[p] :
            somme += s
        for i,(idx,s) in enumerate(Points_w[p]) :
            Points_w[p][i] = (idx,s/somme)

    W_nodes = np.zeros(N,dtype=np.float64)
    for p,liste_faces in enumerate(Points_w) :
        for idx,s in liste_faces :
            W_nodes[idx] += s
    if verbeux :
        print("W_nodes calculé.")
    u = inv[e_u]
    v = inv[e_v]
    W = e_w
    U = np.minimum(u, v)
    V = np.maximum(u, v)
    if verbeux:
        print(f"Arêtes uniques (U<V): {U.size}")
    order = np.argsort(W) # parallel_sort si besoin
    U = U[order]
    V = V[order]
    W = W[order]
    if verbeux :
        print("Arêtes triées.")
    liste_composantes = kruskal(U,V,W,N)
    if verbeux :
        print(f"Kruskal appliqué. Nombre de composantes connexes : {len(liste_composantes)}. Taille de la première : {liste_composantes[0].size}")

    labels_faces = -np.ones(N, dtype=np.int64)
    idx_cluster = 0
    for idx_cc in liste_composantes :
        U_mst = U[idx_cc]
        V_mst = V[idx_cc]
        W_mst = W[idx_cc]
        # On met tous les indices bout à bout
        all_nodes = np.concatenate((U_mst.ravel(), V_mst.ravel()))
    
        # uniques : anciens indices triés
        # inverse : pour chaque entrée de all_nodes, le nouvel indice (0..nb_indices-1)
        uniques, inverse = np.unique(all_nodes, return_inverse=True)
        nb_indices = uniques.size
    
        # On sépare à nouveau pour retrouver la forme de U_mst / V_mst
        M = U_mst.size
        U_new = inverse[:M]
        V_new = inverse[M:]
        W_nodes_cc = W_nodes[uniques]
        Z_cc = condense_tree(W_nodes_cc, U_new, V_new, W_mst, min_cluster_size=min_cluster_size, check_sorted=True) # check_sorted à mettre à False
        if verbeux :
            print(f"condense_tree appliqué. Z_cc = {Z_cc}")
        if splitting is None :
            res = GetClusters(Z_cc, method, splitting=splitting, verbose=verbeux)
        else :
            res = GetClusters(Z_cc, method, splitting=splitting, points=X, Face_to_points=Face_to_points, verbose=verbeux)    
        if verbeux :
            print("GetClusters")
        max_index = -1
        for idx, nodes in enumerate(res['clusters']):
            if idx > max_index :
                max_index = idx
            labels_faces[uniques[nodes]] = idx_cluster + idx
        idx_cluster += max_index +1

    labels_points_multiple = [[] for _ in range(n)]
    for p,liste_faces_w in enumerate(Points_w) :
        clusters = {-1:0.}
        for face,w in liste_faces_w :
            cl = labels_faces[face]
            if cl in clusters :
                clusters[cl] += w
            else :
                clusters[cl] = w
        labels_points_multiple[p] = sorted(clusters.items(), key=lambda x: x[1], reverse=True)

    labels_points_unique = -np.ones(n, dtype=np.int64)
    for p, l_clusters in enumerate(labels_points_multiple) :
        cl = l_clusters[0][0]
        labels_points_unique[p] = cl

    # 4. Label Propagation / Cleanup
    # If subsampling was used, we need to map core labels back to full dataset AND propagate to missing points
    if subsample < 1.0 and not is_sparse_metric:
        # Create full label array
        labels_full = np.full(n_full, -1, dtype=np.int64)
        labels_full[idx_core] = labels_points_unique
        
        # Propagate to non-core points using FAISS/KDTree
        # We query only the non-core points against the core points
        mask_core = np.zeros(n_full, dtype=bool)
        mask_core[idx_core] = True
        X_query = M_full[~mask_core]
        
        if X_query.shape[0] > 0:
            if verbeux:
                print(f"Propagating labels to {X_query.shape[0]} non-core points (k=5)...")
            
            # Use k=5 weighted vote or simple majority?
            # Geometry module's propagate_labels_knn uses majority vote on k neighbors.
            y_pred = propagate_labels_knn(X, labels_points_unique, X_query, k=5, metric=metric)
            labels_full[~mask_core] = y_pred
            
        labels_points_unique = labels_full
        # Note: labels_points_multiple is hard to propagate probabilistically without huge cost. 
        # We leave it as core-only or don't return it for full?
        # If user asked for return_multi_clusters, it's messy. We return it only for core or expanded trivially.
        # For now, let's keep labels_points_multiple aligned with X (core).
        
        # Update X to be full for next step if label_all_points is True?
        # label_all_points typically implies denoising.
        X = M_full

    # 5. Denoising (label_all_points)
    # Replaces old sklearn knn_fill
    if label_all_points and delaunay_possible:
        mask_u = labels_points_unique == -1
        if mask_u.any():
            mask_l = ~mask_u
            if mask_l.any():
                if verbeux:
                    print(f"Denoising {mask_u.sum()} noise points using k-NN...")
                X_train = X[mask_l]
                y_train = labels_points_unique[mask_l]
                X_query = X[mask_u]
                
                # k for denoising? Typically small or min_samples?
                # HDBScan often uses 1-NN for "prediction".
                # Let's use min_samples or 1? Using 1 preserves density boundaries better for noise filling.
                # Using k smooths.
                k_denoise = 1
                y_filled = propagate_labels_knn(X_train, y_train, X_query, k=k_denoise, metric=metric)
                labels_points_unique[mask_u] = y_filled

    if return_multi_clusters:
        if subsample < 1.0 and not is_sparse_metric:
             # Warning: multi-clusters only valid for core indices
             pass
        return labels_points_unique, labels_points_multiple
    return labels_points_unique
