
import numpy as np
from hgp_clusterer import HGPClusterer
from sklearn.datasets import make_blobs

def main():
    # Deux blobs : un dense, un diffus, assez proches
    X1, _ = make_blobs(n_samples=500, centers=[[0, 0]], cluster_std=0.3, random_state=42)
    X2, _ = make_blobs(n_samples=1000, centers=[[1.2, 1.2]], cluster_std=0.6, random_state=42)
    
    # Bruit de fond
    rng = np.random.default_rng(42)
    noise = rng.uniform(low=-2, high=4, size=(200, 2))
    
    X = np.vstack([X1, X2, noise])
    print(f"Dataset: {len(X)} points (500 dense, 1000 diffuse, 200 noise)")
    
    # Run HGP with Delaunay
    # min_cluster_size=20 pour ignorer les petits artefacts du bruit
    print("Running HypergraphPercol...")
    clusterer = HGPClusterer(
        K=2, 
        min_cluster_size=30, 
        min_samples=10, 
        complex_chosen='delaunay', 
        verbose=True, # Logs activés pour debugger
        subsample=1.0
    )
    labels = clusterer.fit_predict(X)
    
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nRésultat final:")
    for lbl, count in zip(unique, counts):
        type_str = "Bruit" if lbl == -1 else f"Cluster {lbl}"
        print(f"  {type_str}: {count} points")

    if len(unique) > 1 and -1 in unique:
        print("\nSuccès: Clusters détectés et bruit identifié.")
    elif len(unique) > 1:
        print("\nSuccès: Clusters détectés (pas de bruit).")
    else:
        print("\nÉchec potentiel: Un seul cluster ou tout bruit.")

if __name__ == "__main__":
    main()
