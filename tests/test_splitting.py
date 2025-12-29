import numpy as np
from hgp_clusterer import HGPClusterer

def test_splitting_simple():
    # 1. Generate some data (2 clusters + noise)
    rng = np.random.default_rng(42)
    X = np.concatenate([
        rng.normal(0, 0.5, (5, 2)),
        rng.normal(5, 0.5, (5, 2)),
        # rng.uniform(-2, 7, (20, 2)) # Noise
    ])
    
    # 2. Define a splitting function that always returns False (stops immediately)
    def my_splitting(parent_idx, children_list_idx):
        print(f"Splitting check: Parent size {len(parent_idx)}, Children count {len(children_list_idx)}")
        # Check types
        assert isinstance(parent_idx, np.ndarray)
        assert isinstance(children_list_idx, list)
        if len(children_list_idx) > 0:
            assert isinstance(children_list_idx[0], np.ndarray)
        return True # Always split to test deep recursion

    # 3. Run HGPClusterer with splitting
    print("Running HGPClusterer with splitting...")
    clusterer = HGPClusterer(
        min_cluster_size=10, 
        epsilon_fusion=0.1, # Trigger batch processing logic
        splitting=my_splitting,
        verbose=True
    )
    labels = clusterer.fit_predict(X)
    
    print("Labels unique:", np.unique(labels))
    print("Test passed!")

if __name__ == "__main__":
    test_splitting_simple()
