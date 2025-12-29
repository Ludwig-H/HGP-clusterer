
import unittest
import numpy as np
import time
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from hgp_clusterer import HypergraphPercol, HGPClusterer

class TestRefactoring(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        n_per_blob = 50
        # Overlapping blobs (Same as splitting demo)
        self.X = np.vstack([
            np.random.normal(0, 0.6, (n_per_blob, 2)),
            np.random.normal(2, 0.6, (n_per_blob, 2))
        ])
        
    def test_legacy_vs_class_simple(self):
        """Check if Function and Class return same labels on standard parameters."""
        print("\nTesting Legacy vs Class (EOM)...")
        
        # Legacy
        t0 = time.time()
        labels_legacy = HypergraphPercol(self.X, min_cluster_size=10, K=2, min_samples=15)
        dt_legacy = time.time() - t0
        
        # Class
        t0 = time.time()
        clusterer = HGPClusterer(min_cluster_size=10, K=2, min_samples=15)
        labels_class = clusterer.fit_predict(self.X)
        dt_class = time.time() - t0
        
        print(f"Time Legacy: {dt_legacy:.4f}s")
        print(f"Time Class:  {dt_class:.4f}s")
        
        # Results should be identical
        np.testing.assert_array_equal(labels_legacy, labels_class)
        
    def test_splitting_optimization(self):
        """Check if Splitting produces correct results (class vs legacy) and runs."""
        print("\nTesting Splitting Rule...")
        
        def split_rule(parent, children):
            if len(parent) > 60:
                return True
            return False
            
        # Legacy
        labels_legacy = HypergraphPercol(
            self.X, min_cluster_size=10, K=2, min_samples=15, 
            splitting=split_rule
        )
        
        # Class
        clusterer = HGPClusterer(
            min_cluster_size=10, K=2, min_samples=15, 
            splitting=split_rule
        )
        labels_class = clusterer.fit_predict(self.X)
        
        unique_l = np.unique(labels_class[labels_class >= 0])
        print(f"Clusters found: {len(unique_l)}")
        self.assertEqual(len(unique_l), 2, "Should split the big component into 2")
        
        np.testing.assert_array_equal(labels_legacy, labels_class)
        
    def test_refine_clusters(self):
        """Test fit once, predict multiple times."""
        print("\nTesting Refine Clusters...")
        clusterer = HGPClusterer(min_cluster_size=10, K=2, min_samples=15)
        clusterer.fit(self.X)
        
        # 1. Default (EOM) -> Should be 1 big cluster (since connected)
        lbl1 = clusterer.labels_
        u1 = np.unique(lbl1[lbl1 >= 0])
        print(f"Pass 1 (EOM): {len(u1)} clusters")
        
        # 2. Refine with Splitting -> Should be 2 clusters
        def split_rule(p, c): return len(p) > 60
        
        t0 = time.time()
        lbl2 = clusterer.refine_clusters(splitting=split_rule)
        dt = time.time() - t0
        u2 = np.unique(lbl2[lbl2 >= 0])
        print(f"Pass 2 (Split): {len(u2)} clusters (Time: {dt:.4f}s)")
        
        self.assertEqual(len(u2), 2)
        self.assertNotEqual(len(u1), len(u2))

if __name__ == '__main__':
    unittest.main()
