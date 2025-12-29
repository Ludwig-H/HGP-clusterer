
import unittest
import numpy as np
from hgp_clusterer import HGPClusterer

class TestMultiClusters(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        # Create 2 blobs
        self.X = np.vstack([
            np.random.normal(0, 0.5, (50, 2)),
            np.random.normal(3, 0.5, (50, 2))
        ])

    def test_return_multi_clusters_init(self):
        """Test initialization with return_multi_clusters."""
        clusterer = HGPClusterer(return_multi_clusters=True)
        self.assertTrue(clusterer.return_multi_clusters)

    def test_fit_predict_multi(self):
        """Test fit_predict returns tuple."""
        clusterer = HGPClusterer(min_cluster_size=10, K=2, return_multi_clusters=True)
        res = clusterer.fit_predict(self.X)
        
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 2)
        labels, multi = res
        
        self.assertEqual(labels.shape[0], 100)
        self.assertEqual(len(multi), 100)
        
        # Check structure of multi
        # list of lists of tuples (cluster_id, proba)
        self.assertIsInstance(multi, list)
        self.assertIsInstance(multi[0], list)
        if len(multi[0]) > 0:
            item = multi[0][0]
            self.assertIsInstance(item, tuple)
            self.assertEqual(len(item), 2)
            
    def test_multi_clusters_attribute(self):
        """Test accessing .multi_clusters_ attribute."""
        clusterer = HGPClusterer(min_cluster_size=10, K=2, return_multi_clusters=True)
        clusterer.fit(self.X)
        
        self.assertIsNotNone(clusterer.multi_clusters_)
        self.assertEqual(len(clusterer.multi_clusters_), 100)

if __name__ == '__main__':
    unittest.main()
