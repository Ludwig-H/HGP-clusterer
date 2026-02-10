# HGP-clusterer

**HGP-clusterer** est une implémentation Python haute performance de l'algorithme de clustering par percolation d'hypergraphes. Il combine la topologie algébrique (complexes simpliciaux) et la théorie de la percolation pour détecter des clusters de formes complexes, même en présence de bruit important.

Ce projet inclut désormais un **double backend géométrique (CGAL et Geogram)** optimisé pour le calcul parallèle intensif.

## Fonctionnalités Clés

- **Clustering Topologique** : Détection robuste de structures non-convexes via l'ordre-k Delaunay.
- **Double Backend Géométrique** :
  - **CGAL** (référence) : Exactitude arithmétique garantie.
  - **Geogram** (expérimental/performant) : Moteur massivement parallèle optimisé pour les machines multicœurs, incluant une implémentation "Zero-Malloc" de l'algorithme de Welzl et une extraction de graphe thread-safe.
- **Raffinement Dynamique** : "Splitting" interactif des clusters sans recalcul géométrique coûteux.
- **Performance** : Cœur C++/Cython/TBB, passage à l'échelle sur des millions de points.

## Installation

### Standard (Recommandé)

```bash
pip install .
```

Pour bénéficier de toutes les fonctionnalités géométriques :

```bash
pip install .[geometry,umap]
```

### Développement

```bash
pip install -e .
```

### Pré-requis système
- **Python** >= 3.9
- **Compilateur C++** (g++ ou clang) compatible C++14.
- **CMake** (pour la compilation du binding géométrique).

L'installation détectera automatiquement la présence de Geogram (dans `geogram_install/`) ou CGAL sur le système.

## Utilisation

La classe `HGPClusterer` suit l'API standard de scikit-learn.

```python
import numpy as np
from hgp_clusterer import HGPClusterer

# Génération de données
X = np.random.RandomState(42).randn(1000, 2)

# Initialisation
# backend='geogram' est recommandé pour les gros jeux de données (N > 100k)
clusterer = HGPClusterer(
    min_cluster_size=20,
    min_samples=5,
    K=2,                  # K=1 pour graphe simple, K=2 pour triangles
    backend='geogram',    # ou 'cgal'
    verbose=True
)

labels = clusterer.fit_predict(X)

print(f"Clusters trouvés : {len(np.unique(labels[labels >= 0]))}")
```

## Optimisations Récentes (Geogram)

Le backend Geogram a été profondément optimisé pour le cas "Order-k Delaunay" en dimension 3 (lifté en 4D) :
1.  **Parallélisme TBB** : L'extraction du dual de Delaunay est entièrement parallélisée.
2.  **Welzl Zero-Malloc** : Calcul des rayons de sphères englobantes sans allocation dynamique (gain de performance majeur).
3.  **Fast 4D Lower-Hull** : Algorithme simplifié pour l'extraction du diagramme de puissance, réduisant l'arithmétique et l'empreinte mémoire.

## Licence

**ATTENTION : USAGE NON-COMMERCIAL UNIQUEMENT.**

Ce logiciel est distribué sous une licence restrictive.
- **Autorisé** : Usage académique, recherche, éducation, projets personnels.
- **Interdit** : Tout usage commercial, intégration dans un produit vendu, services payants.

Voir le fichier `LICENSE` pour le texte complet.

Copyright (c) 2026 Ludwig-H.