# HGP-clusterer

**HGP-clusterer** est une implémentation Python performante de l'algorithme de clustering par percolation d'hypergraphes. Il combine la topologie algébrique (complexes simpliciaux) et la théorie de la percolation pour détecter des clusters de formes complexes, même en présence de bruit important.

L'algorithme suit ces étapes clés :
1.  Construction d'un **hypergraphe** (complexe de Rips ou Delaunay).
2.  Calcul d'un **Arbre Couvrant Minimum (MST)** sur le graphe dual (les faces deviennent des nœuds).
3.  Condensation de l'arbre en une hiérarchie stable (similaire à HDBSCAN).
4.  Sélection des clusters optimaux par **Excess of Mass (EOM)** ou critères de stabilité.

## Installation

### Standard (Recommandé)

```bash
pip install .
```

Pour bénéficier de l'accélération géométrique (sphères minimales via `cyminiball`) et des outils de réduction de dimension (UMAP) :

```bash
pip install .[geometry,umap]
```

### Développement

Pour modifier le code source :

```bash
pip install -e .
```

### Pré-requis système
- **Python** >= 3.9
- **Compilateur C++** (g++ ou clang) pour l'extension Cython.

*(Optionnel)* Pour la filtration exacte "Order-k Delaunay", compilez les binaires CGAL fournis dans `CGALDelaunay/` avec `python scripts/setup_cgal.py`. Sinon, l'algorithme utilise automatiquement l'approximation Rips (très performante).

## Utilisation

La classe `HGPClusterer` suit l'API standard de scikit-learn (`fit`, `predict`).

```python
import numpy as np
from hgp_clusterer import HGPClusterer

# Génération de données
X = np.random.RandomState(42).randn(1000, 2)

# Initialisation et ajustement
clusterer = HGPClusterer(
    min_cluster_size=20,  # Taille minimale d'un cluster
    min_samples=5,        # Paramètre de robustesse au bruit
    K=2,                  # Dimension des simplexes (2 = triangles)
    verbose=True
)

labels = clusterer.fit_predict(X)

print(f"Nombre de clusters trouvés : {len(np.unique(labels[labels >= 0]))}")
```

## Fonctionnalités Avancées

### Raffinement Dynamique (Splitting)

Une force unique de HGP est la capacité de "découper" des clusters connectés par de fins ponts sans recalculer toute la structure géométrique. Vous pouvez définir une règle de découpage personnalisée et ré-extraire les clusters instantanément.

```python
# Après un premier fit()
def ma_regle_de_split(parent_indices, children_list_indices):
    # Exemple : diviser si le parent est trop gros (> 100 points)
    if len(parent_indices) > 100:
        return True
    return False

nouvelles_labels = clusterer.refine_clusters(splitting=ma_regle_de_split)
```

### Gestion de la Mémoire et Performance

Le cœur de l'algorithme est écrit en **Cython** et optimise agressivement l'utilisation mémoire via des graphes duaux implicites et des structures union-find rapides. Il passe à l'échelle sur des millions de points.

## Dépannage

- **ImportError cyminiball** : Si l'installation échoue, le package bascule automatiquement sur une implémentation NumPy (légèrement plus lente mais universelle).
- **Problèmes de compilation** : Assurez-vous que `python-dev` ou `python.h` est accessible. Sur Linux : `sudo apt install python3-dev`.

## Licence

MIT
