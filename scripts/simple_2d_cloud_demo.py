#!/usr/bin/env python
"""Démonstration rapide de HGPClusterer sur un nuage 2D de 20 points."""
from __future__ import annotations

import numpy as np

from hgp_clusterer import HGPClusterer


def make_dataset(seed: int = 8) -> np.ndarray:
    """Construit deux amas gaussiens bien séparés en 2D."""
    rng = np.random.default_rng(seed)
    cloud_a = rng.normal(loc=(-1.5, -1.0), scale=0.25, size=(10, 2))
    cloud_b = rng.normal(loc=(1.5, 1.0), scale=0.25, size=(10, 2))
    return np.vstack([cloud_a, cloud_b])


def main() -> None:
    X = make_dataset()
    clusterer = HGPClusterer(
        K=2,
        min_cluster_size=5,
        min_samples=5,
        complex_chosen="rips",
        verbose=True,
    )
    labels = clusterer.fit_predict(X)
    
    unique, counts = np.unique(labels, return_counts=True)
    print("Labels uniques et effectifs:")
    for label, count in zip(unique, counts):
        print(f"  Cluster {label}: {count} points")


if __name__ == "__main__":
    main()
