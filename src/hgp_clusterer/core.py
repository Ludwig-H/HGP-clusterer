from .union_find import UnionFind
from ._cython import kruskal

import numpy as np

from sklearn.decomposition import PCA

from joblib import Parallel, delayed, cpu_count


def HypergraphPercol(
    M: np.ndarray,
    K: int = 2,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    metric: str = "euclidean",
    method = None,
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
    faces_raw, e_u, e_v, e_w, nS = _build_graph_KSimplexes(
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
    # faces_raw_arr = np.asarray(faces_raw, dtype=np.int64, order="C")
    # e_u_arr = np.asarray(e_u, dtype=np.int64)
    # e_v_arr = np.asarray(e_v, dtype=np.int64)
    # e_w_arr = np.asarray(e_w, dtype=np.float64)
    faces_unique, inv = np.unique(faces_raw, axis=0, return_inverse=True)
    N = faces_unique.shape[0]
    if verbeux:
        print(f"Faces uniques: {N} (compression {faces_raw.shape[0]}→{faces_unique.shape[0]})")
    u = inv[e_u]
    v = inv[e_v]
    W = e_w
    U = np.minimum(u, v)
    V = np.maximum(u, v)
    # order = np.lexsort((vv, uu))
    # uu = uu[order]
    # vv = vv[order]
    # ww = w[order]
    # change = np.r_[True, (uu[1:] != uu[:-1]) | (vv[1:] != vv[:-1])]
    # gidx = np.flatnonzero(change)
    # ww = np.minimum.reduceat(ww, gidx)
    # uu = uu[gidx]
    # vv = vv[gidx]
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
    ### Ici répartir les poids des points sur les faces = (K-1)-simplexes

    for idx_cc in liste_composantes :
        U_mst = U[idx_cc]
        V_mst = V[idx_cc]
        W_mst = W[idx_cc]
        faces_unique.shape[0]
    
    
    
    
    UF_faces = UnionFind(faces_unique.shape[0])
    mst_faces_sorted = _kruskal_mst_from_edges(faces_unique.shape[0], uu, vv, ww, UF_faces)
    if verbeux:
        m = faces_unique.shape[0]
        e_mst = len(mst_faces_sorted)
        comps = max(0, m - e_mst) if m else 0
        print(f"MST faces: {e_mst} arêtes, composantes estimées: {comps}")
    labels_points_unique, labels_points_multiple = build_Z_mst_occurrences_components(
        faces_unique,
        mst_faces_sorted,
        min_cluster_size=min_cluster_size,
        verbose=verbeux,
        distinct_mode="owner",
        DBSCAN_threshold=DBSCAN_threshold,
    )
    labels_points_unique = np.asarray(labels_points_unique)

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
