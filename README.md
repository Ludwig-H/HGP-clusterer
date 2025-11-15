# HGP-clusterer

HypergraphPercol est une implémentation Python du clustering par percolation d'hypergraphes. L'algorithme construit des complexes simpliciaux (ordre-k, Delaunay ou Rips), applique un arbre de Kruskal condensé à la HDBSCAN, puis assigne chaque point au cluster le plus probable.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

L'installation crée une extension Cython (`hgp_clusterer._cython`). Les dépendances de base couvrent `gudhi`, `hdbscan` et `scikit-learn`. Pour bénéficier de la détection de sphères minimales ultra précise, installez en plus `cyminiball` via l'extra `geometry` (facultatif sur Python ≥3.12 grâce au repli NumPy intégré) :

```bash
pip install -e .[geometry]
```

### CGAL (optionnel)

Pour activer la filtration « order-k Delaunay », compilez les binaires CGAL fournis dans `CGALDelaunay/` via `python scripts/setup_cgal.py`. Sans CGAL, l'API se replie automatiquement sur la filtration Rips/GUDHI.

## Utilisation rapide

```python
import numpy as np
from hgp_clusterer import HypergraphPercol

X = np.random.RandomState(0).randn(200, 3)
labels = HypergraphPercol(
    X,
    K=2,
    metric="euclidean",
    complex_chosen="rips",   # "auto" choisit une filtration adaptée
    min_cluster_size=15,
    label_all_points=True,
    verbeux=True,
)
print(np.unique(labels))
```

Paramètres utiles :

- `K` : dimension des simplexes (2 ⇒ triangles).
- `min_cluster_size` / `min_samples` : contrôle la condensation HDBSCAN.
- `complex_chosen` : `"auto"`, `"rips"`, `"delaunay"` ou `"orderk_delaunay"`.
- `metric` : `"euclidean"`, `"precomputed"` ou `"sparse"` (triplets `(i, j, d)`).
- `weight_face` : pondération des points sur les (K-1)-faces (`"lambda"` par défaut).
- `label_all_points` : comble les points bruit via k-NN pondéré.

`HypergraphPercol` retourne par défaut les étiquettes majoritaires. Passez `return_multi_clusters=True` pour récupérer, pour chaque point, la distribution pondérée des clusters atteints.

## Dépannage

1. **ImportError cyminiball** : un repli NumPy est désormais inclus. Pour de meilleures performances, installez l'extra `geometry`.
2. **Binaire CGAL manquant** : l'API bascule automatiquement sur la filtration Rips. Compilez CGAL si vous souhaitez l'ordre-k exact.
3. **Installation lente** : assurez-vous d'avoir un compilateur C++17 (`g++`).

## Tests rapides

Après installation, un mini test permet de vérifier l'API :

```bash
python - <<'PY'
import numpy as np
from hgp_clusterer import HypergraphPercol
X = np.random.RandomState(0).randn(20, 3)
print(HypergraphPercol(X, K=2, complex_chosen='rips'))
PY
```

### Exemple 2D (20 points)

Le script `scripts/simple_2d_cloud_demo.py` génère deux amas gaussiens en 2D (10 points chacun) et exécute `HypergraphPercol` avec `K=2` et `min_cluster_size=5`. Lancez-le pour reproduire le test demandé :

```bash
python scripts/simple_2d_cloud_demo.py
```
