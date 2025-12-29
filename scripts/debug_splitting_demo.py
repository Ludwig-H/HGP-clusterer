"""
Demonstration script for the 'splitting' callback in HypergraphPercol.

This script shows how to inject custom domain logic to refine clusters
after the standard selection (EOM/Leaf).
"""
import sys
import os
import numpy as np

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hgp_clusterer import HypergraphPercol

def demo_splitting():
    print("=== HypergraphPercol Splitting Callback Demo ===")
    
    # 1. Generate Data: Two overlapping blobs
    # They form a single connected component in the graph, but have distinct density centers.
    np.random.seed(42)
    n_per_blob = 50
    # Blob 1 at (0,0), Blob 2 at (2,0). Sigma=0.6 implies significant overlap.
    X = np.vstack([
        np.random.normal(0, 0.6, (n_per_blob, 2)),
        np.random.normal(2, 0.6, (n_per_blob, 2))
    ])
    
    print(f"Data: {len(X)} points generated (2 overlapping Gaussian blobs).")

    # 2. Define the Splitting Rule
    # This function is called recursively top-down on selected clusters.
    # Return True to force a split (discard parent, keep children).
    # Return False to keep the parent.
    def max_size_splitting_rule(parent_idx, children_idx_list):
        n_parent = len(parent_idx)
        
        # Example Business Logic: "No cluster should be larger than 60 points."
        # If a cluster is too big, we try to split it into its sub-components.
        MAX_SIZE = 60
        
        print(f"[Callback] Checking cluster of size {n_parent}...")
        
        if n_parent > MAX_SIZE:
            if not children_idx_list:
                print(f"  -> Too big ({n_parent} > {MAX_SIZE}), but it's a leaf (no children). Cannot split.")
                return False
            
            print(f"  -> Too big ({n_parent} > {MAX_SIZE}). Decision: SPLIT into {len(children_idx_list)} children.")
            return True
        
        print(f"  -> Size OK ({n_parent} <= {MAX_SIZE}). Decision: KEEP.")
        return False

    # 3. Run Clustering
    # We use a large min_samples to ensure the graph connects the two blobs initially.
    print("\n--- Running Clustering with Splitting Rule ---")
    labels = HypergraphPercol(
        X,
        min_cluster_size=10,
        K=2,
        min_samples=15, 
        metric="euclidean",
        splitting=max_size_splitting_rule,
        verbeux=False # Set to True to see algorithm internals
    )

    # 4. Analyze Results
    unique_labels = np.unique(labels[labels >= 0])
    print("\n--- Final Results ---")
    print(f"Total Clusters Found: {len(unique_labels)}")
    for lbl in unique_labels:
        count = np.sum(labels == lbl)
        print(f"  Cluster {lbl}: {count} points")

    if len(unique_labels) == 2:
        print("\nSUCCESS: The huge component was successfully split into 2 smaller clusters.")
    else:
        print(f"\nRESULT: Found {len(unique_labels)} clusters.")

if __name__ == "__main__":
    demo_splitting()