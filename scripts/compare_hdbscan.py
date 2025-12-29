#!/usr/bin/env python
"""Comparaison rapide HGP-Clusterer (subsample) vs HDBSCAN classique."""
import time
import numpy as np
try:
    from sklearn.cluster import HDBSCAN
except ImportError:
    # Fallback for older sklearn
    try:
        import hdbscan
        # Wrap hdbscan.HDBSCAN if needed, but usually it has same API
        class HDBSCAN(hdbscan.HDBSCAN):
             pass
    except ImportError:
        print("HDBSCAN not installed.")
        HDBSCAN = None

from hgp_clusterer import HGPClusterer
from sklearn.metrics import adjusted_rand_score

def make_dataset(n_samples=1500, seed=42):
    rng = np.random.default_rng(seed)
    # 3 clusters gaussiens 3D
    c1 = rng.normal(loc=(0, 0, 0), scale=0.5, size=(n_samples//3, 3))
    c2 = rng.normal(loc=(3, 3, 0), scale=0.5, size=(n_samples//3, 3))
    c3 = rng.normal(loc=(0, 3, 3), scale=0.5, size=(n_samples//3, 3))
    X = np.vstack([c1, c2, c3])
    y = np.concatenate([np.zeros(len(c1)), np.ones(len(c2)), np.full(len(c3), 2)])
    # Bruit
    noise = rng.uniform(low=-2, high=5, size=(n_samples//10, 3))
    X = np.vstack([X, noise])
    y = np.concatenate([y, np.full(len(noise), -1)])
    return X, y

def main():
    X, y_true = make_dataset()
    print(f"Dataset: {len(X)} points, 3 clusters + bruit")

    # 1. HDBSCAN Classique (Sklearn implementation)
    print("\n--- HDBSCAN (Sklearn) ---")
    labels_hdb = np.full(len(X), -1)
    if HDBSCAN is not None:
        start = time.time()
        try:
            # Sklearn HDBSCAN uses min_samples directly
            clusterer = HDBSCAN(min_cluster_size=30, min_samples=10)
            labels_hdb = clusterer.fit_predict(X)
        except Exception as e:
            print(f"HDBSCAN skipped due to error: {e}")
            
        end = time.time()
        n_clus_hdb = len(set(labels_hdb)) - (1 if -1 in labels_hdb else 0)
        print(f"Temps: {end-start:.4f}s | Clusters: {n_clus_hdb}")
    else:
        print("HDBSCAN not available.")
    
    # 2. HGP-Clusterer (Plein)
    print("\n--- HGP (Full - Auto/Delaunay) ---")
    start = time.time()
    # Utilisation de 'auto' qui choisira Delaunay (ou Rips optimisé) selon la dispo
    clusterer = HGPClusterer(
        min_cluster_size=30, min_samples=10, 
        complex_chosen='delaunay', subsample=1.0, verbose=False
    )
    labels_hgp = clusterer.fit_predict(X)
        
    end = time.time()
    n_clus_hgp = len(set(labels_hgp)) - (1 if -1 in labels_hgp else 0)
    print(f"Temps: {end-start:.4f}s | Clusters: {n_clus_hgp}")

    # 3. HGP-Clusterer (Subsample 20%)
    print("\n--- HGP (Subsample 0.2) ---")
    start = time.time()
    clusterer_sub = HGPClusterer(
        min_cluster_size=30, min_samples=10, 
        complex_chosen='delaunay', subsample=0.2, verbose=True
    )
    labels_hgp_sub = clusterer_sub.fit_predict(X)

    end = time.time()
    n_clus_sub = len(set(labels_hgp_sub)) - (1 if -1 in labels_hgp_sub else 0)
    print(f"Temps: {end-start:.4f}s | Clusters: {n_clus_sub}")
    
    # Scores
    ari_hdb = adjusted_rand_score(y_true, labels_hdb)
    ari_hgp = adjusted_rand_score(y_true, labels_hgp)
    ari_sub = adjusted_rand_score(y_true, labels_hgp_sub)
    
    print("\n--- Scores (ARI vs Ground Truth) ---")
    print(f"HDBSCAN: {ari_hdb:.4f}")
    print(f"HGP Full: {ari_hgp:.4f}")
    print(f"HGP Sub (0.2): {ari_sub:.4f}")

if __name__ == "__main__":
    main()
