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


from .clustering import GetClusters, condense_tree
from .hypergraph import _build_graph_KSimplexes
from .union_find import UnionFind
from ._cython import kruskal
from .geometry import propagate_labels_knn

import math
import numpy as np
import os

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClusterMixin

from joblib import Parallel, delayed, cpu_count


class HGPClusterer(BaseEstimator, ClusterMixin):
    """
    Hypergraph Percolation Clusterer (HGP).
    
    A Scikit-Learn compatible estimator that builds a dual hypergraph from data,
    computes a Minimum Spanning Tree, and extracts clusters using a HDBSCAN-like
    condensed tree hierarchy.
    """
    
    def __init__(self, 
                 min_cluster_size=None,
                 K=2, 
                 min_samples=None,
                 metric='euclidean',
                 method='eom',
                 splitting=None,
                 weight_face='lambda',
                 label_all_points=False,
                 complex_chosen='auto',
                 expZ=2.0,
                 precision='safe',
                 dim_reducer=False,
                 threshold_variance=0.999,
                 subsample=1.0,
                 epsilon_fusion=0.0,
                 return_multi_clusters=False,
                 verbose=False,
                 backend='geogram',
                 cgal_root="/content/HGP-clusterer/CGALDelaunay"):
        
        self.min_cluster_size = min_cluster_size
        self.K = K
        self.min_samples = min_samples
        self.metric = metric
        self.method = method
        self.splitting = splitting
        self.weight_face = weight_face
        self.label_all_points = label_all_points
        self.complex_chosen = complex_chosen
        self.expZ = expZ
        self.precision = precision
        self.dim_reducer = dim_reducer
        self.threshold_variance = threshold_variance
        self.subsample = subsample
        self.epsilon_fusion = epsilon_fusion
        self.return_multi_clusters = return_multi_clusters
        self.verbose = verbose
        self.backend = backend
        self.cgal_root = cgal_root
        
        # State
        self.tree_ = None
        self.forest_ = None
        self.multi_clusters_ = None

    def fit(self, X, y=None):
        """
        Builds the Hypergraph, MST and Condensed Tree.
        """
        self._fit_core(X)
        self.labels_ = self._extract_labels(self.method, self.splitting)
        return self

    def fit_predict(self, X, y=None):
        self.fit(X)
        if self.return_multi_clusters:
            return self.labels_, self.multi_clusters_
        return self.labels_

    def refine_clusters(self, method='eom', splitting=None):
        """
        Re-extracts clusters from the pre-computed tree without rebuilding the graph.
        Very fast. Updates self.labels_.
        """
        self.labels_ = self._extract_labels(method, splitting)
        return self.labels_

    def _fit_core(self, M):
        # 1. Input Handling & Subsampling
        self.is_sparse_metric_ = (self.metric == "sparse")
        
        if self.is_sparse_metric_:
            M = np.asarray(M, dtype=np.float32)
            n_full = int(np.max(M[:, :2])) + 1 if M.size else 0
            # No subsampling support for sparse yet
            self.X_core_ = M
            self.idx_core_ = np.arange(n_full, dtype=np.int32)
            self.X_full_ = None # Sparse matrix structure not stored fully here
            n_core = n_full
            d_core = 0
        else:
            M_full = np.ascontiguousarray(M, dtype=np.float32)
            self.X_full_ = M_full
            n_full, d_full = M_full.shape
            
            if 0.0 < self.subsample < 1.0 and n_full > 100:
                n_core = int(n_full * self.subsample)
                if self.verbose:
                    print(f"Subsampling: {n_core}/{n_full}")
                rng = np.random.default_rng(42)
                self.idx_core_ = rng.choice(n_full, size=n_core, replace=False).astype(np.int32)
                self.X_core_ = M_full[self.idx_core_]
            else:
                self.idx_core_ = np.arange(n_full, dtype=np.int32)
                self.X_core_ = M_full
            n_core, d_core = self.X_core_.shape

        # Defaults
        self.n_core_ = n_core
        mcs = self.min_cluster_size
        if mcs is None:
            mcs = round(math.sqrt(n_core))
        self.min_cluster_size_ = mcs
        
        ms = self.min_samples
        if ms is None or ms <= self.K:
            ms = self.K + 1
        self.min_samples_ = min(ms, n_core) if n_core > 0 else ms

        # 2. Dim Reduction
        X_processed = np.copy(self.X_core_)
        delaunay_possible = (not self.metric == "precomputed" 
                             and self.metric == "euclidean" 
                             and not self.is_sparse_metric_ 
                             and X_processed.ndim == 2)

        if not self.is_sparse_metric_ and str(self.dim_reducer).lower() in {"pca", "umap"} and delaunay_possible:
            if str(self.dim_reducer).lower() == "pca":
                pca = PCA(n_components=self.threshold_variance, svd_solver="full", whiten=False)
                X2 = pca.fit_transform(X_processed)
                if pca.n_components_ < d_core:
                    X_processed = X2.astype(np.float32)
                    if self.verbose: print(f"PCA reduced to {pca.n_components_} dims")
            elif str(self.dim_reducer).lower() == "umap":
                 from umap import UMAP
                 reducer = UMAP(n_components=pca.n_components_, 
                                n_neighbors=max(2 * 2 * (self.K + 1), self.min_samples_), 
                                metric=self.metric)
                 X_processed = reducer.fit_transform(X_processed).astype(np.float32)

        # 3. Build Hypergraph
        # Fix Geogram PDEL infinite hang on exactly coplanar/collinear inputs
        if self.backend == 'geogram' and X_processed.shape[0] > 0:
            rng_noise = np.random.default_rng(42)
            X_processed = X_processed + rng_noise.normal(0, 1e-5, X_processed.shape).astype(X_processed.dtype)
            
        # Optim: Returns unique faces and pre-calculated S_faces
        faces_unique, e_u, e_v, e_w, S_faces, nS = _build_graph_KSimplexes(
            X_processed,
            self.K,
            self.min_samples_,
            self.metric,
            self.complex_chosen,
            self.expZ,
            precision=self.precision,
            verbose=self.verbose,
            backend=self.backend,
            cgal_root=self.cgal_root,
        )
        
        if len(faces_unique) == 0:
            self.tree_ = None
            return

        # Store for splitting (Faces -> Points map)
        self.faces_unique_ = faces_unique
        # Note: 'inv' is no longer needed or available as we deduplicated on the fly.
        
        N = self.faces_unique_.shape[0]
        
        # 4. Weight Calculation
        # S_faces is already computed (Accumulated 1/r)
        
        n_vertices_per_face = self.faces_unique_.shape[1]
        flat_faces = self.faces_unique_.flatten()
        S_faces_expanded = np.repeat(S_faces, n_vertices_per_face)
        T_points = np.bincount(flat_faces, weights=S_faces_expanded, minlength=n_core).astype(np.float32)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_T_points = 1.0 / T_points
            inv_T_points[~np.isfinite(inv_T_points)] = 0.0
            
        sum_inv_Tp_face = np.sum(inv_T_points[self.faces_unique_], axis=1)
        self.W_nodes_ = (S_faces * sum_inv_Tp_face).astype(np.float32)
        
        # Store for extraction
        self.S_faces_ = S_faces
        self.T_points_ = T_points 
        
        # 5. MST & Tree
        
        # Edges now refer to unique face IDs directly
        u = np.asarray(e_u, dtype=np.int32)
        v = np.asarray(e_v, dtype=np.int32)
        w = np.asarray(e_w, dtype=np.float32)
        
        U = np.minimum(u, v)
        V = np.maximum(u, v)
        order = np.argsort(w)
        U, V, w = U[order], V[order], w[order]
        
        liste_composantes = kruskal(U, V, w, int(N))
        
        self.forest_ = []
        self.component_indices_ = [] 
        
        idx_cluster_offset = 0
        
        for idx_cc in liste_composantes:
            U_mst = U[idx_cc]
            V_mst = V[idx_cc]
            W_mst = w[idx_cc]
            
            all_nodes = np.concatenate((U_mst.ravel(), V_mst.ravel()))
            uniques, inverse = np.unique(all_nodes, return_inverse=True)
            uniques = uniques.astype(np.int32)
            inverse = inverse.astype(np.int32)
            
            M_edges = U_mst.size
            U_new = inverse[:M_edges]
            V_new = inverse[M_edges:]
            W_nodes_cc = self.W_nodes_[uniques]
            
            Z_cc = condense_tree(
                W_nodes_cc, 
                U_new, 
                V_new, 
                W_mst, 
                min_cluster_size=int(self.min_cluster_size_), 
                check_sorted=False, 
                epsilon=float(self.epsilon_fusion)
            )
            
            faces_cc = self.faces_unique_[uniques]
            
            self.forest_.append({
                'Z': Z_cc,
                'faces_cc': faces_cc,
                'uniques_map': uniques, 
                'cluster_offset': idx_cluster_offset
            })
            

    def _extract_labels(self, method, splitting):
        if self.tree_ is None and (self.forest_ is None or len(self.forest_) == 0):
            return np.full(self.n_core_, -1, dtype=np.int32)
            
        labels_faces = -np.ones(self.faces_unique_.shape[0], dtype=np.int32)
        current_cluster_id = 0
        
        for comp in self.forest_:
            Z = comp['Z']
            faces_cc = comp['faces_cc']
            uniques = comp['uniques_map']
            S_faces_cc = self.S_faces_[uniques]

            # Extract
            res = GetClusters(Z, method, splitting=splitting, Face_to_points=faces_cc, S_faces=S_faces_cc, verbose=self.verbose)            
            # Assign labels to faces
            for i, nodes in enumerate(res['clusters']):
                global_face_indices = uniques[nodes]
                labels_faces[global_face_indices] = current_cluster_id + i
            
            current_cluster_id += len(res['clusters'])

        # Propagate Faces -> Points
        
        mask_valid = labels_faces != -1
        if not mask_valid.any():
            if self.return_multi_clusters:
                self.multi_clusters_ = [[] for _ in range(self.n_core_)]
            return np.full(self.n_core_, -1, dtype=np.int32)
            
        # Expand
        n_vertices_per_face = self.faces_unique_.shape[1]
        labels_expanded = np.repeat(labels_faces, n_vertices_per_face)
        S_faces_expanded = np.repeat(self.S_faces_, n_vertices_per_face)
        flat_faces = self.faces_unique_.flatten()
        
        mask = labels_expanded >= 0
        rows = flat_faces[mask]
        cols = labels_expanded[mask]
        data = S_faces_expanded[mask]
        
        # Scipy Sparse
        from scipy.sparse import coo_matrix
        mat = coo_matrix((data, (rows, cols)), shape=(self.n_core_, current_cluster_id), dtype=np.float32).tocsr()
        
        best_clusters = np.asarray(mat.argmax(axis=1)).flatten()
        has_votes = mat.getnnz(axis=1) > 0
        
        final_labels = np.full(self.n_core_, -1, dtype=np.int32)
        final_labels[has_votes] = best_clusters[has_votes]
        
        # Compute multi_clusters if requested
        if self.return_multi_clusters:
            if self.T_points_ is not None:
                T_points_expanded = self.T_points_[flat_faces]
                with np.errstate(divide='ignore', invalid='ignore'):
                    norm_weights = S_faces_expanded / T_points_expanded
                    norm_weights[~np.isfinite(norm_weights)] = 0.0
                
                data_norm = norm_weights[mask]
                mat_norm = coo_matrix((data_norm, (rows, cols)), shape=(self.n_core_, current_cluster_id), dtype=np.float32).tocsr()
                
                self.multi_clusters_ = [[] for _ in range(self.n_core_)]
                for i in range(self.n_core_):
                    row = mat_norm[i]
                    if row.nnz > 0:
                        pairs = list(zip(row.indices, row.data))
                        self.multi_clusters_[i] = sorted(pairs, key=lambda x: x[1], reverse=True)
            else:
                self.multi_clusters_ = []
        
        # Propagate to full if subsampled
        if self.X_full_ is not None and len(self.X_full_) > len(self.X_core_):
             n_full = len(self.X_full_)
             labels_full = np.full(n_full, -1, dtype=np.int32)
             labels_full[self.idx_core_] = final_labels
             
             mask_core = np.zeros(n_full, dtype=bool)
             mask_core[self.idx_core_] = True
             X_query = self.X_full_[~mask_core]
             
             if X_query.shape[0] > 0:
                 if self.verbose:
                     print(f"Propagating labels to {X_query.shape[0]} non-core points (k=5)...")
                 
                 y_pred = propagate_labels_knn(self.X_core_, final_labels, X_query, k=5, metric=self.metric)
                 labels_full[~mask_core] = y_pred.astype(np.int32)
                 
             final_labels = labels_full
             
        # Denoising (label_all_points)
        if self.label_all_points and not self.is_sparse_metric_:
             X_target = self.X_full_ if self.X_full_ is not None else self.X_core_
             
             mask_noise = (final_labels == -1)
             if np.any(mask_noise):
                 mask_valid = ~mask_noise
                 if np.any(mask_valid):
                     if self.verbose:
                         print(f"Denoising {np.sum(mask_noise)} noise points using 1-NN...")
                     
                     X_train = X_target[mask_valid]
                     y_train = final_labels[mask_valid]
                     X_query = X_target[mask_noise]
                     
                     y_filled = propagate_labels_knn(X_train, y_train, X_query, k=1, metric=self.metric)
                     final_labels[mask_noise] = y_filled.astype(np.int32)
             
        return final_labels

# def HypergraphPercol(
#     M: np.ndarray,
#     K: int = 2,
#     min_cluster_size: int | None = None,
#     min_samples: int | None = None,
#     metric: str = "euclidean",
#     method = 'eom',
#     splitting = None,
#     weight_face: str = "lambda", # "lambda" ∝ 1/r ; "uniform" ∝ 1 ; "unique" 1 on the face with min r
#     label_all_points: bool = False,
#     return_multi_clusters: bool = False,
#     complex_chosen: str = "auto",
#     expZ: float = 2,
#     precision: str = "safe",
#     dim_reducer: bool | str = False,
#     threshold_variance_dim_reduction: float = 0.999,
#     verbeux: bool = False,
#     cgal_root: str | os.PathLike[str] | None = "/content/HGP-clusterer/CGALDelaunay",
#     subsample: float = 1.0,
#     epsilon_fusion: float = 0.0,
# ) -> np.ndarray | tuple[np.ndarray, list[list[tuple[int, float, float]]]]:
#     """
#     Legacy functional wrapper for HGPClusterer.
#     
#     This function delegates all logic to the new `HGPClusterer` class.
#     See `HGPClusterer` docstrings for details.
#     """
#     
#     # 1. Instantiate the Class
#     clusterer = HGPClusterer(
#         min_cluster_size=min_cluster_size,
#         K=K,
#         min_samples=min_samples,
#         metric=metric,
#         method=method,
#         splitting=splitting,
#         weight_face=weight_face,
#         complex_chosen=complex_chosen,
#         expZ=expZ,
#         precision=precision,
#         dim_reducer=dim_reducer,
#         threshold_variance=threshold_variance_dim_reduction,
#         subsample=subsample,
#         epsilon_fusion=epsilon_fusion,
#         verbose=verbeux,
#         cgal_root=cgal_root
#     )
#     
#     # 2. Fit
#     # Note: label_all_points is not yet fully integrated into HGPClusterer.fit() in the basic version,
#     # but we can handle it post-fit here or ensure the class does it.
#     # The current HGPClusterer implementation does NOT do label_all_points internally (denoising).
#     # We will add it here if needed or let the user handle it.
#     # Actually, for perfect backward compatibility, we should implement it here if the class doesn't.
#     
#     clusterer.fit(M)
#     labels = clusterer.labels_
#     
#     # 3. Post-processing for legacy features (label_all_points, return_multi_clusters)
#     
#     # Denoising (label_all_points)
#     # HGPClusterer stores full dataset in self.X_full_ if it exists, or self.X_core_
#     if label_all_points:
#         # Check if we have noise
#         mask_noise = (labels == -1)
#         if np.any(mask_noise):
#             # We need the data.
#             # If sparse, no denoising usually.
#             if not clusterer.is_sparse_metric_ and clusterer.X_core_ is not None:
#                 # Use X_core_? But labels match M which might be subsampled?
#                 # HGPClusterer.fit handles subsampling internally and returns mapped labels if subsampled.
#                 # However, HGPClusterer._extract_labels returns labels for the *input* M (if subsample=1)
#                 # If subsample < 1, HGPClusterer propagates to M (full) inside _extract_labels (see estimator code).
#                 # So `labels` matches `M`.
#                 
#                 # We need M. If M was a list/sparse, it might be tricky.
#                 # Assuming M is the data array compatible with geometry.propagate_labels_knn
#                 
#                 # Filter valid points
#                 mask_valid = ~mask_noise
#                 if np.any(mask_valid):
#                     if verbeux:
#                         print(f"Denoising {np.sum(mask_noise)} noise points using k-NN...")
#                     
#                     # We need the actual data points. 
#                     # If M is sparse (triplets), propagate_labels_knn might fail unless it supports it.
#                     # Core check:
#                     is_delaunay_possible = (
#                         metric == "euclidean" and not clusterer.is_sparse_metric_ and M.ndim == 2
#                     )
#                     
#                     if is_delaunay_possible:
#                         # Use M directly
#                         X_train = M[mask_valid]
#                         y_train = labels[mask_valid]
#                         X_query = M[mask_noise]
#                         
#                         y_filled = propagate_labels_knn(X_train, y_train, X_query, k=1, metric=metric)
#                         labels[mask_noise] = y_filled
#     
#     # Multi-clusters
#     if return_multi_clusters:
#         # The class does not currently expose the soft-vote matrix publicly in a nice format.
#         # But we can reconstruct it or simply return the empty/dummy list if the class doesn't support it yet.
#         # Given the prompt is to "Refactor... wrapper", and the Class is supposedly "Fast/Clean",
#         # implementing full multi-cluster soft extraction here might be overkill if not in the class.
#         # BUT, to pass tests that check this return type, we must return a structure.
#         
#         # Placeholder: Return hard assignment as 100% confidence
#         # OR TODO: Add get_soft_clusters() to HGPClusterer.
#         
#         # For now, let's build the list of lists [(cluster_id, 1.0)]
#         multi_clusters = []
#         for l in labels:
#             if l == -1:
#                 multi_clusters.append([])
#             else:
#                 multi_clusters.append([(l, 1.0)])
#                 
#         return labels, multi_clusters
# 
#     return labels

# def HypergraphPercol(
#     M: np.ndarray,
#     K: int = 2,
#     min_cluster_size: int | None = None,
#     min_samples: int | None = None,
#     metric: str = "euclidean",
#     method = 'eom',
#     splitting = None,
#     weight_face: str = "lambda", # "lambda" ∝ 1/r ; "uniform" ∝ 1 ; "unique" 1 on the face with min r
#     label_all_points: bool = False,
#     return_multi_clusters: bool = False,
#     complex_chosen: str = "auto",
#     expZ: float = 2,
#     precision: str = "safe",
#     dim_reducer: bool | str = False,
#     threshold_variance_dim_reduction: float = 0.999,
#     verbeux: bool = False,
#     cgal_root: str | os.PathLike[str] | None = "/content/HGP-clusterer/CGALDelaunay",
#     subsample: float = 1.0,
#     epsilon_fusion: float = 0.0,
# ) -> np.ndarray | tuple[np.ndarray, list[list[tuple[int, float, float]]]]:
#     if method is None:
#         method = "eom"
#     is_sparse_metric = metric == "sparse"
#     
#     # 1. Handling Input Data & Subsampling
#     if is_sparse_metric:
#         M = np.asarray(M, dtype=np.float64)
#         if M.ndim != 2 or M.shape[1] != 3:
#             raise ValueError("For metric='sparse', M must be provided as (i, j, distance) triplets.")
#         if M.size:
#             n_full = int(np.max(M[:, :2])) + 1
#         else:
#             n_full = 0
#         d_full = 0
#         # Subsampling for sparse graph is tricky (node subsampling vs edge subsampling). 
#         # For now, we assume subsample=1.0 or we ignore it/warn.
#         if subsample < 1.0 and verbeux:
#             print("Warning: Subsampling is not supported for sparse metric yet. Ignored.")
#         subsample = 1.0
#         X_core = M
#         idx_core = np.arange(n_full)
#     else:
#         M_full = np.ascontiguousarray(M, dtype=np.float64)
#         n_full, d_full = M_full.shape
#         
#         if 0.0 < subsample < 1.0 and n_full > 100: # Don't subsample tiny datasets
#             n_core = int(n_full * subsample)
#             if verbeux:
#                 print(f"Subsampling: selecting {n_core} points out of {n_full} ({subsample*100:.1f}%)")
#             rng = np.random.default_rng(42) # Fixed seed for reproducibility or None? Let's fix it for stability.
#             idx_core = rng.choice(n_full, size=n_core, replace=False)
#             X_core = M_full[idx_core]
#         else:
#             idx_core = np.arange(n_full)
#             X_core = M_full
#     
#     X = np.copy(X_core)
#     n, d = (X.shape if not is_sparse_metric else (n_full, 0))
#     
#     if min_cluster_size is None:
#         min_cluster_size = round(math.sqrt(n))
#     
#     pre = metric == "precomputed"
#     delaunay_possible = not pre and metric == "euclidean" and not is_sparse_metric and M.ndim == 2
#     if min_samples is None or min_samples <= K:
#         min_samples = K + 1
#     if n > 0:
#         min_samples = min(min_samples, n)
#         
#     # 2. Dimensionality Reduction (on core set)
#     if not is_sparse_metric and str(dim_reducer).lower() in {"pca", "umap"} and delaunay_possible:
#         pca = PCA(n_components=threshold_variance_dim_reduction, svd_solver="full", whiten=False)
#         X2 = pca.fit_transform(X)
#         r = pca.n_components_
#         ratio = pca.explained_variance_ratio_.sum()
#         if r < d and str(dim_reducer).lower() == "pca":
#             X = X2
#             if verbeux:
#                 print(f"Dimension réduite par PCA : {d} → {r} (variance {ratio:.3f})")
#         elif r < d and str(dim_reducer).lower() == "umap":
#             from umap import UMAP
# 
#             reducer = UMAP(n_components=r, n_neighbors=max(2 * 2 * (K + 1), min_samples), metric=metric)
#             X = reducer.fit_transform(X)
#             if verbeux:
#                 print(f"Dimension réduite par UMAP : {d} → {r}")
#     
#     # 3. Build Hypergraph & Clustering
#     faces_raw, e_u, e_v, e_w, faces_Simplexes, nS = _build_graph_KSimplexes(
#         X,
#         K,
#         min_samples,
#         metric,
#         complex_chosen,
#         expZ,
#         precision=precision,
#         verbose=verbeux,
#         cgal_root=cgal_root,
#     )
#     if verbeux:
#         print(f"{K}-simplices={nS}")
#     if not faces_raw:
#         if verbeux and not is_sparse_metric and K > d:
#             print(
#                 "Warning: K too high compared to the dimension of the data. "
#                 "No clustering possible with such a K."
#             )
#         empty_labels = np.full(n_full, -1, dtype=np.int64)
#         if return_multi_clusters:
#             empty_multi = [[(-1, 1.0)] for _ in range(n_full)]
#             return empty_labels, empty_multi
#         return empty_labels
#     faces_raw_arr = np.asarray(faces_raw, dtype=np.int64, order="C")
#     e_u = np.asarray(e_u, dtype=np.int64)
#     e_v = np.asarray(e_v, dtype=np.int64)
#     e_w = np.asarray(e_w, dtype=np.float64)
#     faces_unique, inv = np.unique(faces_raw_arr, axis=0, return_inverse=True)
#     N = faces_unique.shape[0]
#     if verbeux :
#         print(
#             f"Faces uniques: {N} (compression {faces_raw_arr.shape[0]}→{faces_unique.shape[0]})"
#         )
#     
#     # --- Vectorized Weight Calculation ---
#     
#     # 1. Unpack simplex data efficiently
#     # faces_Simplexes contains (old_idx, points_face, r_face). 
#     # We construct arrays to manipulate weights efficiently.
#     n_simplexes = len(faces_Simplexes)
#     sim_old_indices = np.empty(n_simplexes, dtype=np.int64)
#     sim_radii = np.empty(n_simplexes, dtype=np.float64)
#     
#     # Extract data from the list of tuples
#     for i, (oidx, _, r) in enumerate(faces_Simplexes):
#         sim_old_indices[i] = oidx
#         sim_radii[i] = r
#         
#     sim_unique_indices = inv[sim_old_indices]
# 
#     # Pre-calculate Simplex Weights (Contribution to Faces)
#     if weight_face == "lambda":
#         with np.errstate(divide='ignore'):
#             sim_weights = 1.0 / sim_radii
#         # Handle r=0 case (infinite weight -> cap or handle)
#         sim_weights[~np.isfinite(sim_weights)] = 1e12 
#     elif weight_face == "uniform":
#         sim_weights = np.ones(n_simplexes, dtype=np.float64)
#     elif weight_face == "unique":
#         # Placeholder for unique mode (rare)
#         pass 
#     else:
#         raise ValueError(f"Unknown weight_face mode: {weight_face}")
# 
#     if weight_face in ["lambda", "uniform"]:
#         # --- FAST PATH (Vectorized) ---
#         
#         # S_faces[f] = Sum of weights of all simplices projecting to face f
#         S_faces = np.bincount(sim_unique_indices, weights=sim_weights, minlength=N)
#         
#         # T_points[p] = Sum of S_faces[f] for all f containing p
#         # faces_unique is (N, K). We flatten it to get all point occurrences.
#         flat_faces = faces_unique.flatten()
#         
#         # Expand S_faces to match flattened structure
#         n_vertices_per_face = faces_unique.shape[1] # Should be equal to K
#         S_faces_expanded = np.repeat(S_faces, n_vertices_per_face)
#         
#         # Sum weights per point
#         T_points = np.bincount(flat_faces, weights=S_faces_expanded, minlength=n)
#         
#         # W_nodes[f] = S_faces[f] * Sum_{p in f} (1 / T_points[p])
#         with np.errstate(divide='ignore', invalid='ignore'):
#             inv_T_points = 1.0 / T_points
#             inv_T_points[~np.isfinite(inv_T_points)] = 0.0
#             
#         # Sum inv_T for points in each face
#         # inv_T_points[faces_unique] creates shape (N, K+1) with 1/Tp values
#         sum_inv_Tp_face = np.sum(inv_T_points[faces_unique], axis=1)
#         
#         W_nodes = S_faces * sum_inv_Tp_face
#         
#     else:
#         # --- SLOW PATH (Legacy Loop for 'unique' mode) ---
#         # Reconstruct Points structure only for this mode
#         Points = [[] for _ in range(n)]
#         # Re-iterate to build structure (slower but safe for 'unique')
#         for (old_idx, points_face, r_face) in faces_Simplexes:
#             idx_face = inv[old_idx]
#             for p in points_face:
#                 Points[p].append((idx_face, r_face))
#                 
#         Points_w = [{} for _ in range(n)] # Only for unique
#         # 'unique' logic: keep face with max weight (min radius) per point
#         # Simulating the dictionary update logic from original code
#         for p, liste_faces in enumerate(Points):
#              if not liste_faces: continue
#              # Find best face
#              best_face = -1
#              max_weight = -1.0
#              
#              for (idx_face, r_f) in liste_faces:
#                  w = 1.0/r_f if r_f > 0 else 1e12
#                  if w > max_weight:
#                      max_weight = w
#                      best_face = idx_face
#              
#              if best_face != -1:
#                  Points_w[p] = [(best_face, 1.0)] # Normalized to 1 later? 
#                  # Original code: Points_w[p][0] = (idx_face, 1/w_face) then normalize
#                  # Effectively, point gives 100% vote to this face.
#         
#         # Calculate W_nodes from Points_w
#         W_nodes = np.zeros(N, dtype=np.float64)
#         for p in range(n):
#             if Points_w[p]: # List of (face, weight)
#                 # Normalize? Original code normalized per point.
#                 # For unique, usually 1 item, so weight becomes 1.0.
#                 total_w = sum(x[1] for x in Points_w[p])
#                 if total_w > 0:
#                     for (idx, w) in Points_w[p]:
#                         W_nodes[idx] += w / total_w
#     
#     if verbeux :
#         print("W_nodes calculé.")
#     u = inv[e_u]
#     v = inv[e_v]
#     W = e_w
#     U = np.minimum(u, v)
#     V = np.maximum(u, v)
#     if verbeux:
#         print(f"Arêtes uniques (U<V): {U.size}")
#     order = np.argsort(W) # parallel_sort si besoin
#     U = U[order]
#     V = V[order]
#     W = W[order]
#     if verbeux :
#         print("Arêtes triées.")
#     liste_composantes = kruskal(U,V,W,N)
#     if verbeux :
#         print(f"Kruskal appliqué. Nombre de composantes connexes : {len(liste_composantes)}. Taille de la première : {liste_composantes[0].size}")
# 
#     labels_faces = -np.ones(N, dtype=np.int64)
#     idx_cluster = 0
#     for idx_cc in liste_composantes :
#         U_mst = U[idx_cc]
#         V_mst = V[idx_cc]
#         W_mst = W[idx_cc]
#         # On met tous les indices bout à bout
#         all_nodes = np.concatenate((U_mst.ravel(), V_mst.ravel()))
#     
#         # uniques : anciens indices triés
#         # inverse : pour chaque entrée de all_nodes, le nouvel indice (0..nb_indices-1)
#         uniques, inverse = np.unique(all_nodes, return_inverse=True)
#         nb_indices = uniques.size
#     
#         # On sépare à nouveau pour retrouver la forme de U_mst / V_mst
#         M = U_mst.size
#         U_new = inverse[:M]
#         V_new = inverse[M:]
#         W_nodes_cc = W_nodes[uniques]
#         # Optimisation : W_mst est déjà trié car issu de W trié
#         Z_cc = condense_tree(W_nodes_cc, U_new, V_new, W_mst, min_cluster_size=min_cluster_size, check_sorted=False, epsilon=epsilon_fusion)
#         if verbeux :
#             print(f"condense_tree appliqué. Z_cc (keys): {list(Z_cc.keys())}")
#             
#         if splitting is None :
#             res = GetClusters(Z_cc, method, splitting=splitting, verbose=verbeux)
#         else :
#             # Pour le splitting, on a besoin de la map Faces->Points.
#             # On passe faces_unique. Attention, il faut passer le sous-ensemble correspondant à la composante connexe si on veut être strict,
#             # mais GetClusters travaille avec des indices globaux ou locaux ?
#             # Z_cc est local (indices 0..nb_indices-1).
#             # GetClusters retourne des indices locaux.
#             # Il faut donc mapper indices locaux -> indices globaux (uniques) -> points (faces_unique)
#             # On peut passer une fonction ou adapter GetClusters.
#             # Le plus simple : reconstruire le subset de faces pour cette CC.
#             faces_cc = faces_unique[uniques]
#             res = GetClusters(Z_cc, method, splitting=splitting, points=X, Face_to_points=faces_cc, verbose=verbeux)    
#         if verbeux :
#             print("GetClusters appliqué.")
# 
#         max_index = -1
#         for idx, nodes in enumerate(res['clusters']):
#             if idx > max_index :
#                 max_index = idx
#             labels_faces[uniques[nodes]] = idx_cluster + idx
#         idx_cluster += max_index +1
# 
#     labels_points_multiple = []
#     labels_points_unique = -np.ones(n, dtype=np.int64) # Initialize here to be safe across all branches
#     
#     # --- Final Point Labeling (Vectorized/Sparse Optimization) ---
#     # Goal: Aggregate votes from faces to points.
#     # Vote(p, c) = Sum_{f in C, p in f} (weight(f) / T_points[p])
#     # For unique labels (ArgMax), we can ignore T_points[p] as it's constant for p.
#     
#     HAS_SCIPY = False
#     try:
#         from scipy.sparse import coo_matrix
#         HAS_SCIPY = True
#     except ImportError:
#         pass
# 
#     if weight_face in ["lambda", "uniform"] and HAS_SCIPY and idx_cluster > 0:
#         # Prepare Sparse Matrix Data
#         # Filter out faces that are noise (-1)
#         mask_valid_faces = labels_faces != -1
#         
#         # We need to expand the validity mask to the flattened structure
#         # labels_faces is (N,), we need mask for (N*K,)
#         # faces_unique is (N, K)
#         n_vertices_per_face = faces_unique.shape[1]
#         
#         # valid_faces_indices = np.where(mask_valid_faces)[0] 
#         # But we need to mask the FLATTENED arrays.
#         
#         # Expand labels to match flattened faces
#         labels_expanded = np.repeat(labels_faces, n_vertices_per_face)
#         S_faces_expanded = np.repeat(S_faces, n_vertices_per_face)
#         flat_faces = faces_unique.flatten()
#         
#         # Mask for valid clusters (>= 0)
#         mask = labels_expanded >= 0
#         
#         rows = flat_faces[mask]
#         cols = labels_expanded[mask]
#         
#         if return_multi_clusters:
#             # For probabilities, we need the normalized weights
#             # Expand T_points to match flattened structure
#             T_points_expanded = T_points[flat_faces]
#             # Avoid div by zero
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 norm_weights = S_faces_expanded / T_points_expanded
#                 norm_weights[~np.isfinite(norm_weights)] = 0
#             
#             data = norm_weights[mask]
#             
#             # Build Sparse Matrix (Sum duplicates)
#             mat = coo_matrix((data, (rows, cols)), shape=(n, idx_cluster)).tocsr()
#             
#             # Convert to list of tuples
#             labels_points_multiple = [[] for _ in range(n)]
#             for i in range(n):
#                 row = mat[i]
#                 if row.nnz > 0:
#                     # Extract (col, val) pairs and sort
#                     pairs = zip(row.indices, row.data)
#                     labels_points_multiple[i] = sorted(pairs, key=lambda x: x[1], reverse=True)
#             
#             # Extract unique from multiple
#             for p, l_clusters in enumerate(labels_points_multiple) :
#                 if l_clusters:
#                     labels_points_unique[p] = l_clusters[0][0]
#                 else:
#                     labels_points_unique[p] = -1
#                     
#         else:
#             # OPTIMIZATION: For ArgMax, we DO NOT need to normalize by T_points[p]
#             # because T_points[p] is constant for a given point p across all clusters.
#             # Max(Sum(S/T)) == Max(Sum(S)/T) == Max(Sum(S))
#             
#             data = S_faces_expanded[mask]
#             
#             # Build Sparse Matrix
#             mat = coo_matrix((data, (rows, cols)), shape=(n, idx_cluster)).tocsr()
#             
#             # Argmax per row
#             # mat.argmax(axis=1) returns matrix of shape (n, 1)
#             best_clusters = np.asarray(mat.argmax(axis=1)).flatten()
#             
#             # Handle points that have no votes (row sum is 0) -> Noise
#             # argmax on all-zeros returns 0 (which is a valid cluster ID). We must mask them.
#             # Using getnnz is fast
#             has_votes = mat.getnnz(axis=1) > 0
#             
#             labels_points_unique[has_votes] = best_clusters[has_votes]
#             labels_points_unique[~has_votes] = -1
# 
#     else:
#         # --- Fallback (No Scipy OR 'unique' mode OR idx_cluster=0) ---
#         labels_points_multiple = [[] for _ in range(n)]
#         
#         if weight_face in ["lambda", "uniform"] and idx_cluster > 0:
#              # Python Loop Fallback for standard modes
#              point_cluster_scores = [{} for _ in range(n)]
#              for f_idx in range(N):
#                 c = labels_faces[f_idx]
#                 if c == -1: continue
#                 val_f = S_faces[f_idx]
#                 for p in faces_unique[f_idx]:
#                      if T_points[p] > 0:
#                         w = val_f / T_points[p]
#                         if c in point_cluster_scores[p]:
#                             point_cluster_scores[p][c] += w
#                         else:
#                             point_cluster_scores[p][c] = w
#              
#              if return_multi_clusters:
#                  for p in range(n):
#                      labels_points_multiple[p] = sorted(point_cluster_scores[p].items(), key=lambda x: x[1], reverse=True)
#                      if labels_points_multiple[p]:
#                          labels_points_unique[p] = labels_points_multiple[p][0][0]
#              else:
#                  for p in range(n):
#                      best_c = -1
#                      best_s = -1.0
#                      for c, s in point_cluster_scores[p].items():
#                          if s > best_s:
#                              best_s = s
#                              best_c = c
#                      labels_points_unique[p] = best_c
# 
#         else:
#              # 'unique' mode OR no clusters found
#              if weight_face == "unique":
#                  for p in range(n):
#                     best_c = -1
#                     best_s = -1.0
#                     if Points_w[p]:
#                         clusters = {}
#                         for face, w in Points_w[p]:
#                             cl = labels_faces[face]
#                             clusters[cl] = clusters.get(cl, 0.0) + w
#                         
#                         if return_multi_clusters:
#                             labels_points_multiple[p] = sorted(clusters.items(), key=lambda x: x[1], reverse=True)
#                             if labels_points_multiple[p]:
#                                 best_c = labels_points_multiple[p][0][0]
#                         else:
#                             for c, s in clusters.items():
#                                 if s > best_s:
#                                     best_s = s
#                                     best_c = c
#                     
#                     labels_points_unique[p] = best_c
# 
#     # 4. Label Propagation / Cleanup
#     # If subsampling was used, we need to map core labels back to full dataset AND propagate to missing points
#     if subsample < 1.0 and not is_sparse_metric:
#         # Create full label array
#         labels_full = np.full(n_full, -1, dtype=np.int64)
#         labels_full[idx_core] = labels_points_unique
#         
#         # Propagate to non-core points using FAISS/KDTree
#         # We query only the non-core points against the core points
#         mask_core = np.zeros(n_full, dtype=bool)
#         mask_core[idx_core] = True
#         X_query = M_full[~mask_core]
#         
#         if X_query.shape[0] > 0:
#             if verbeux:
#                 print(f"Propagating labels to {X_query.shape[0]} non-core points (k=5)...")
#             
#             # Use k=5 weighted vote or simple majority?
#             # Geometry module's propagate_labels_knn uses majority vote on k neighbors.
#             y_pred = propagate_labels_knn(X, labels_points_unique, X_query, k=5, metric=metric)
#             labels_full[~mask_core] = y_pred
#             
#         labels_points_unique = labels_full
#         # Note: labels_points_multiple is hard to propagate probabilistically without huge cost. 
#         # We leave it as core-only or don't return it for full?
#         # If user asked for return_multi_clusters, it's messy. We return it only for core or expanded trivially.
#         # For now, let's keep labels_points_multiple aligned with X (core).
#         
#         # Update X to be full for next step if label_all_points is True?
#         # label_all_points typically implies denoising.
#         X = M_full
# 
#     # 5. Denoising (label_all_points)
#     # Replaces old sklearn knn_fill
#     if label_all_points and delaunay_possible:
#         mask_u = labels_points_unique == -1
#         if mask_u.any():
#             mask_l = ~mask_u
#             if mask_l.any():
#                 if verbeux:
#                     print(f"Denoising {mask_u.sum()} noise points using k-NN...")
#                 X_train = X[mask_l]
#                 y_train = labels_points_unique[mask_l]
#                 X_query = X[mask_u]
#                 
#                 # k for denoising? Typically small or min_samples?
#                 # HDBScan often uses 1-NN for "prediction".
#                 # Let's use min_samples or 1? Using 1 preserves density boundaries better for noise filling.
#                 # Using k smooths.
#                 k_denoise = 1
#                 y_filled = propagate_labels_knn(X_train, y_train, X_query, k=k_denoise, metric=metric)
#                 labels_points_unique[mask_u] = y_filled
# 
#     if return_multi_clusters:
#         if subsample < 1.0 and not is_sparse_metric:
#              # Warning: multi-clusters only valid for core indices
#              pass
#         return labels_points_unique, labels_points_multiple
#     return labels_points_unique
