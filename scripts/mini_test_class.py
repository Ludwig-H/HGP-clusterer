
import sys
import os
import numpy as np

# Ensure we use the local src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hgp_clusterer import HGPClusterer

def mini_test():
    print("=== Mini-Test: HGPClusterer with Splitting ===")
    
    # 1. Data: 2 overlapping blobs (Chain)
    # A single connected component that we want to split by size
    np.random.seed(42)
    X = np.vstack([
        np.random.normal(0, 0.6, (50, 2)),
        np.random.normal(2, 0.6, (50, 2))
    ])
    print(f"Data: {len(X)} points.")

    # 2. Splitting Rule
    def split_if_too_big(parent_idx, children_idx_list):
        # If cluster > 60 points, split it!
        if len(parent_idx) > 60:
            return True
        return False

    # 3. Instantiate Class
    # Note: subsample=1.0 to ensure we get all points
    model = HGPClusterer(
        min_cluster_size=10, 
        K=2, 
        min_samples=15, 
        splitting=split_if_too_big,
        verbose=False
    )

    # 4. Run Fit & Predict
    print("Running fit_predict...")
    try:
        labels = model.fit_predict(X)
    except Exception as e:
        print(f"FAILED with error: {e}")
        raise e

    # 5. Check Results
    unique_labels = np.unique(labels[labels >= 0])
    n_clusters = len(unique_labels)
    print(f"Result: Found {n_clusters} clusters.")
    
    if n_clusters == 2:
        print("SUCCESS: The large component was split into 2.")
    else:
        print(f"FAILURE: Expected 2 clusters, got {n_clusters}.")

if __name__ == "__main__":
    mini_test()
