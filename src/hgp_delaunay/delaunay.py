import numpy as np
from sklearn.base import BaseEstimator

# Try to import internal modules
try:
    from hgp_clusterer.hypergraph import _build_graph_KSimplexes
except ImportError:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from hgp_clusterer.hypergraph import _build_graph_KSimplexes

class HGPDelaunay(BaseEstimator):
    """
    Hypergraph Percolation Delaunay (HGP-Delaunay).
    
    A simpler version of HGP-Clusterer and HGP-Reducer that solely builds the 
    graph of (K-1)-simplexes connected by K-simplexes (with radius r) from the
    Order-K Delaunay mosaic.
    """
    
    def __init__(self,
                 K=2,
                 expZ=1.0,
                 precision='safe',
                 verbose=False,
                 backend='cgal',
                 cgal_root="/content/HGP-clusterer/CGALDelaunay",
                 device='auto'):
        self.K = K
        self.expZ = expZ
        self.precision = precision
        self.verbose = verbose
        self.backend = backend
        self.cgal_root = cgal_root
        self.device = device
        
        # Fixed parameters
        self.metric = 'euclidean'
        self.complex_chosen = 'orderk_delaunay'
        
    def fit(self, X, y=None):
        """
        Builds the Order-K Delaunay graph of (K-1)-simplexes.
        """
        self.fit_transform(X)
        return self
        
    def fit_transform(self, X, y=None):
        """
        Builds the Order-K Delaunay graph of (K-1)-simplexes and returns its components.
        
        Returns
        -------
        faces_unique_ : array-like
            The unique (K-1)-simplexes (faces).
        edges_u_ : array-like
            The source nodes of the graph (indices in faces_unique_).
        edges_v_ : array-like
            The target nodes of the graph (indices in faces_unique_).
        edges_weights_ : array-like
            The filtration weights (radius r^expZ) of the K-simplex connecting edges_u_ and edges_v_.
        """
        X = np.asarray(X)
        
        if self.verbose:
            print(f"Building {self.complex_chosen} graph...")
            
        # Fix Geogram PDEL infinite hang on exactly coplanar/collinear inputs
        X_processed = np.copy(X)
        if self.backend == 'geogram' and X_processed.shape[0] > 0:
            rng_noise = np.random.default_rng(42)
            X_processed = X_processed + rng_noise.normal(0, 1e-5, X_processed.shape).astype(X_processed.dtype)
            
        min_samples = self.K + 1
            
        # Build the graph of K-simplexes
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
        
        # Store attributes
        self.faces_unique_ = faces_unique
        self.edges_u_ = e_u
        self.edges_v_ = e_v
        self.edges_weights_ = e_w
        self.S_faces_ = S_faces
        self.n_unique_ = n_unique
        
        return (self.faces_unique_, self.edges_u_, self.edges_v_, self.edges_weights_)
