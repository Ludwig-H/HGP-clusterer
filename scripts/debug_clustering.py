
import numpy as np
from hgp_clusterer import HGPClusterer
from hgp_clusterer.clustering import condense_tree, GetClusters, _build_dfs_structure

def make_dataset(n_samples=300, seed=42):
    rng = np.random.default_rng(seed)
    c1 = rng.normal(loc=(0, 0), scale=0.3, size=(n_samples//3, 2))
    c2 = rng.normal(loc=(2, 2), scale=0.3, size=(n_samples//3, 2))
    c3 = rng.normal(loc=(-2, 2), scale=0.3, size=(n_samples//3, 2))
    X = np.vstack([c1, c2, c3])
    return X

def debug_run():
    X = make_dataset()
    print(f"Dataset shape: {X.shape}")
    
    # Run HGP
    clusterer = HGPClusterer(
        K=2, 
        min_cluster_size=5, 
        min_samples=3,
        complex_chosen='orderk_delaunay', # Force Order-K Delaunay logic
        verbose=True
    )
    labels = clusterer.fit_predict(X)
    
    unique_labels = np.unique(labels)
    print(f"Labels found: {unique_labels}")
    
if __name__ == "__main__":
    debug_run()
