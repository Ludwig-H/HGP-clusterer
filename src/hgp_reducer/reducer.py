import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# Try to import internal modules
try:
    from hgp_clusterer.hypergraph import _build_graph_KSimplexes
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from hgp_clusterer.hypergraph import _build_graph_KSimplexes

class HGPReducer(BaseEstimator, TransformerMixin):
    """
    Hypergraph Percolation Reducer (HGP-Reducer).
    
    A Scikit-Learn compatible estimator for dimensionality reduction.
    It builds a dual hypergraph from data using Order-K Delaunay triangulation,
    computes the Laplacian of the dual graph (on (K-1)-simplices), finds its 
    spectral embedding, and projects it back to the original points via 
    barycentric prolongation.
    """
    
    def __init__(self,
                 n_components=None,
                 d=None,
                 K=2,
                 min_samples=None,
                 metric='euclidean',
                 weight_face='lambda',
                 complex_chosen='orderk_delaunay',
                 expZ=2.0,
                 laplacian_type='combinatorial',
                 precision='safe',
                 verbose=False,
                 backend='geogram',
                 cgal_root="/content/HGP-clusterer/CGALDelaunay",
                 device='auto'):
        self.n_components = n_components
        self.d = d
        self.K = K
        self.min_samples = min_samples
        self.metric = metric
        self.weight_face = weight_face
        self.complex_chosen = complex_chosen
        self.expZ = expZ
        self.laplacian_type = laplacian_type
        self.precision = precision
        self.verbose = verbose
        self.backend = backend
        self.cgal_root = cgal_root
        self.device = device
        
    def fit(self, X, y=None):
        self.fit_transform(X)
        return self
        
    def fit_transform(self, X, y=None):
        X = np.asarray(X)
        n_points = X.shape[0]
        
        # Determine target dimension
        d_target = self.n_components if self.n_components is not None else self.d
        d_target = d_target if d_target is not None else self.K
        
        min_samples = self.min_samples
        if min_samples is None or min_samples <= self.K:
            min_samples = self.K + 1
            
        if self.verbose:
            print(f"Building {self.complex_chosen} graph...")
            
        # Fix Geogram PDEL infinite hang on exactly coplanar/collinear inputs
        X_processed = np.copy(X)
        if self.backend == 'geogram' and X_processed.shape[0] > 0:
            rng_noise = np.random.default_rng(42)
            X_processed = X_processed + rng_noise.normal(0, 1e-5, X_processed.shape).astype(X_processed.dtype)
            
        # Build the graph of K-simplexes and extract the dual graph representation
        faces_unique, e_u, e_v, e_w, S_faces, n_unique = _build_graph_KSimplexes(
            X_processed,
            self.K,
            min_samples,
            self.metric,
            self.complex_chosen,
            self.expZ,
            precision=self.precision,
            verbose=self.verbose,
            backend=self.backend,
            cgal_root=self.cgal_root,
        )
        
        n_faces = len(faces_unique)
        if n_faces == 0:
            if self.verbose:
                print("No faces found, returning zeros.")
            self.embedding_ = np.zeros((n_points, d_target))
            return self.embedding_
            
        if self.verbose:
            print(f"Graph built with {n_faces} faces. Constructing adjacency matrix W...")
            
        # Adjacency matrix W construction
        # e_w is the filtration weight r^expZ of the K-simplex connecting e_u and e_v.
        # We set edge weights to 1 / r^expZ as requested.
        with np.errstate(divide='ignore', invalid='ignore'):
            # Use max(1e-12) to avoid Division by Zero on perfectly zero radii
            edge_weights = 1.0 / np.maximum(e_w, 1e-12)
        
        # Build sparse matrix W (coo format for fast construction)
        W = sp.coo_matrix((edge_weights, (e_u, e_v)), shape=(n_faces, n_faces))
        # Symetrise since the connectivity is undirected
        W = W + W.T
        W_coo = W.tocoo()
        D_diag = np.asarray(W_coo.sum(axis=1)).flatten()
        
        # Build LA Trick Matrix (M) completely on CPU via fast COO arrays to bypass all SpGEMM
        if self.laplacian_type == 'normalized':
            D_inv_sqrt = np.zeros_like(D_diag)
            mask = D_diag > 0
            D_inv_sqrt[mask] = 1.0 / np.sqrt(D_diag[mask])
            
            # M = D^{-1/2} W D^{-1/2} (computed element-wise on edges in O(N))
            M_data = W_coo.data * D_inv_sqrt[W_coo.row] * D_inv_sqrt[W_coo.col]
            M_cpu = sp.coo_matrix((M_data, (W_coo.row, W_coo.col)), shape=(n_faces, n_faces)).tocsr()
        else:
            c_bound = 2.0 * np.max(D_diag) + 0.1
            diag_indices = np.arange(n_faces)
            diag_data = c_bound - D_diag
            
            M_data = np.concatenate([W_coo.data, diag_data])
            M_row = np.concatenate([W_coo.row, diag_indices])
            M_col = np.concatenate([W_coo.col, diag_indices])
            M_cpu = sp.coo_matrix((M_data, (M_row, M_col)), shape=(n_faces, n_faces)).tocsr()
            
        # Hardware acceleration check
        use_cuda = False
        if self.device in ['auto', 'cuda']:
            try:
                import cupy as cp
                import cupyx.scipy.sparse as csp
                import cupyx.scipy.sparse.linalg as csplinalg
                use_cuda = True
            except ImportError:
                if self.device == 'cuda' and self.verbose:
                    print("Warning: CUDA requested but cupy/cupyx is not installed. Falling back to CPU.")
                use_cuda = False
                
        if self.verbose:
            print(f"Constructing {self.laplacian_type} Laplacian (CUDA: {use_cuda})...")

        U = None
        # Compute d_target + 1 eigenvectors so we can discard the first one (corresponding to eigenvalue 0)
        k_eig = min(d_target + 1, n_faces - 2)
        if k_eig <= 0:
            U = np.zeros((n_faces, d_target))
        else:
            if use_cuda:
                # --- CUDA Fast Path ---
                try:
                    if self.verbose:
                        print(f"Computing {d_target} eigenvectors on GPU (using SpGEMM-free LA Trick)...")
                    
                    M_gpu = csp.csr_matrix(M_cpu)
                    evals_gpu, evecs_gpu = csplinalg.eigsh(M_gpu, k=k_eig, which='LA')
                    
                    if self.laplacian_type == 'normalized':
                        evals = 1.0 - cp.asnumpy(evals_gpu)
                        # Rescale to get Random Walk Laplacian eigenvectors: u = D^{-1/2} v
                        evecs = D_inv_sqrt[:, None] * cp.asnumpy(evecs_gpu)
                    else:
                        evals = c_bound - cp.asnumpy(evals_gpu)
                        evecs = cp.asnumpy(evecs_gpu)
                    
                    # Sort explicitly to be certain
                    idx = np.argsort(evals)
                    evecs = evecs[:, idx]
                    U = evecs[:, 1:d_target+1] # Skip the 1st
                except Exception as e:
                    if self.verbose:
                        print(f"GPU eigsh failed ({e}), falling back to CPU...")
                    use_cuda = False # Trigger fallback

            if not use_cuda:
                # --- CPU Path ---
                if self.verbose:
                    print(f"Computing {d_target} eigenvectors on CPU (using LA Trick)...")
                    
                try:
                    evals_M, evecs_M = splinalg.eigsh(M_cpu, k=k_eig, which='LA')
                except Exception as e:
                    if self.verbose:
                        print(f"eigsh LA failed ({e}), falling back to LM...")
                    evals_M, evecs_M = splinalg.eigsh(M_cpu, k=k_eig, which='LM')
                
                if self.laplacian_type == 'normalized':
                    evals = 1.0 - evals_M
                    # Rescale to get Random Walk Laplacian eigenvectors: u = D^{-1/2} v
                    evecs = D_inv_sqrt[:, None] * evecs_M
                else:
                    evals = c_bound - evals_M
                    evecs = evecs_M
                
                # Sort explicitly
                idx = np.argsort(evals)
                evecs = evecs[:, idx]
                U = evecs[:, 1:d_target+1] # Skip the 1st
        
        # Edge case: If n_faces was very small (e.g., less than d_target+2) pad with zeros
        if U.shape[1] < d_target:
            pad = np.zeros((n_faces, d_target - U.shape[1]))
            U = np.hstack([U, pad])
            
        if self.verbose:
            print("Eigenvectors computed. Constructing barycentric prolongation matrix P...")
            
        # Barycentric prolongation
        # We reuse the specific weighting calculation exactly as performed in HGP-Clusterer.
        n_vertices_per_face = faces_unique.shape[1]
        flat_faces = faces_unique.flatten()
        
        # S_faces contains sum of weights of K-simplices projecting on each face.
        S_faces_expanded = np.repeat(S_faces, n_vertices_per_face)
        # T_points[p] is the total sum of face weights surrounding point p.
        T_points = np.bincount(flat_faces, weights=S_faces_expanded, minlength=n_points)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_T_points = 1.0 / T_points
            inv_T_points[~np.isfinite(inv_T_points)] = 0.0
            
        # Weight formula: P_{p, f} = S_faces[f] / T_points[p]
        # Equivalently: P_{p, f} = S_faces[f] * inv_T_points[p]
        P_data = S_faces_expanded * inv_T_points[flat_faces]
        faces_indices = np.repeat(np.arange(n_faces), n_vertices_per_face)
        
        P = sp.coo_matrix((P_data, (flat_faces, faces_indices)), shape=(n_points, n_faces)).tocsr()
        
        if self.verbose:
            print("Projecting dimensions from faces back to points...")
            
        # Multiply our operator P with eigenvectors U to embed points
        self.embedding_ = P.dot(U)
        
        if self.verbose:
            print("HGP-Reducer fit_transform complete.")
            
        return self.embedding_