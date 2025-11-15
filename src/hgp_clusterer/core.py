from .clustering import GetClusters, condense_tree
from .hypergraph import _build_graph_KSimplexes
from .union_find import UnionFind
from ._cython import kruskal

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
    cgal_root: str | os.PathLike[str] | None = "/content/HypergraphPercol/CGALDelaunay",
) -> np.ndarray | tuple[np.ndarray, list[list[tuple[int, float, float]]]]:
    if method is None:
        method = "eom"
    is_sparse_metric = metric == "sparse"
    if is_sparse_metric:
        M = np.asarray(M, dtype=np.float64)
        if M.ndim != 2 or M.shape[1] != 3:
            raise ValueError("For metric='sparse', M must be provided as (i, j, distance) triplets.")
        if M.size:
            n = int(np.max(M[:, :2])) + 1
        else:
            n = 0
        d = 0
    else:
        M = np.ascontiguousarray(M, dtype=np.float64)
        n, d = M.shape
    if min_cluster_size is None:
        min_cluster_size = round(math.sqrt(n))
    X = np.copy(M)
    pre = metric == "precomputed"
    delaunay_possible = not pre and metric == "euclidean" and not is_sparse_metric and M.ndim == 2
    if min_samples is None or min_samples <= K:
        min_samples = K + 1
    if n > 0:
        min_samples = min(min_samples, n)
    if not is_sparse_metric and str(dim_reducer).lower() in {"pca", "umap"} and delaunay_possible:
        pca = PCA(n_components=threshold_variance_dim_reduction, svd_solver="full", whiten=False)
        X2 = pca.fit_transform(M)
        r = pca.n_components_
        ratio = pca.explained_variance_ratio_.sum()
        if r < d and str(dim_reducer).lower() == "pca":
            X = X2
            if verbeux:
                print(f"Dimension réduite par PCA : {d} → {r} (variance {ratio:.3f})")
        elif r < d and str(dim_reducer).lower() == "umap":
            from umap import UMAP

            reducer = UMAP(n_components=r, n_neighbors=max(2 * 2 * (K + 1), min_samples), metric=metric)
            X = reducer.fit_transform(M)
            if verbeux:
                print(f"Dimension réduite par UMAP : {d} → {r}")
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
    # if not faces_raw:
    #     if not is_sparse_metric and K > d:
    #         print("Warning: K too high compared to the dimension of the data. No clustering possible with such a K.")
    #     if return_multi_clusters:
    #         return np.full(n, -1, dtype=np.int64), [(-1, 1.0, 1.0)] * n
    #     return np.full(n, 0, dtype=np.int64)
    faces_raw_arr = np.asarray(faces_raw, dtype=np.int64, order="C")
    e_u = np.asarray(e_u, dtype=np.int64)
    e_v = np.asarray(e_v, dtype=np.int64)
    e_w = np.asarray(e_w, dtype=np.float64)
    faces_unique, inv = np.unique(faces_raw_arr, axis=0, return_inverse=True)
    N = faces_unique.shape[0]
    if verbeux :
        print(f"Faces uniques: {N} (compression {faces_raw.shape[0]}→{faces_unique.shape[0]})")
    
    ### Ici répartir les poids des points sur les faces = (K-1)-simplexes
    Points = [[] for _ in range(n)]
    for (old_idx, points_face, r_face) in faces_Simplexes :
        idx_face = inv[old_idx]
        for p in points_face :
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
        print(f"Kruskal appliqué. Nombre de composantes connexes : {len(liste_composantes)}")

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
        res = GetClusters(Z_cc, method, splitting=splitting, verbose=verbeux)
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

    def knn_fill_weighted(X_data: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        from sklearn.neighbors import KNeighborsClassifier

        X_data = np.asarray(X_data)
        y = labels.copy()
        mask_u = y == -1
        if not mask_u.any():
            return y
        mask_l = ~mask_u
        if not mask_l.any():
            return y
        k = min(k, int(mask_l.sum()))
        clf = KNeighborsClassifier(n_neighbors=k, weights="distance", n_jobs=-1)
        clf.fit(X_data[mask_l], y[mask_l])
        y[mask_u] = clf.predict(X_data[mask_u])
        return y

    if label_all_points and delaunay_possible:
        labels_points_unique = knn_fill_weighted(M, labels_points_unique, min_samples)
    if return_multi_clusters:
        return labels_points_unique, labels_points_multiple
    return labels_points_unique
