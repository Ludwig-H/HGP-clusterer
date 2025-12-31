import numpy as np
import time
from hgp_clusterer.delaunay import orderk_delaunay3
from hgp_clusterer.geometry import minimum_enclosing_ball

def verify_radii():
    print("--- Verification des rayons calculés par C++ vs Python ---")
    
    # 1. Générer des données
    N = 100
    dim = 2
    K = 2
    points = np.random.rand(N, dim).astype(np.float64)
    
    print(f"Points: {N}, Dimension: {dim}, K: {K}")
    
    # 2. Appel C++ (Nouveau Binding)
    try:
        start = time.time()
        # Cela retourne maintenant (simplices, weights)
        simplices, cxx_weights = orderk_delaunay3(points, K, verbose=False)
        end = time.time()
        print(f"C++ terminé en {end - start:.4f}s. {len(simplices)} simplexes trouvés.")
    except ImportError:
        print("Erreur: L'extension C++ n'est pas installée/compilée.")
        return
    except Exception as e:
        print(f"Erreur d'exécution C++: {e}")
        return

    if len(simplices) == 0:
        print("Aucun simplexe retourné.")
        return

    # 3. Vérification croisée avec Python (Cyminiball/Welzl)
    print("Vérification sur un échantillon de 10 simplexes...")
    
    indices_to_check = np.random.choice(len(simplices), min(10, len(simplices)), replace=False)
    
    max_error = 0.0
    
    for idx in indices_to_check:
        simplex_indices = simplices[idx]
        simplex_points = points[simplex_indices]
        
        # Calcul Python de référence
        _, py_radius_sq = minimum_enclosing_ball(simplex_points)
        
        # Valeur C++
        cxx_radius_sq = cxx_weights[idx]
        
        diff = abs(py_radius_sq - cxx_radius_sq)
        max_error = max(max_error, diff)
        
        print(f"  Simplex {idx}: Py={py_radius_sq:.6f}, C++={cxx_radius_sq:.6f} | Diff={diff:.2e}")

    print(f"\nErreur maximale observée: {max_error:.2e}")
    if max_error < 1e-5:
        print("✅ SUCCÈS : Les calculs C++ correspondent à la référence Python.")
    else:
        print("❌ ÉCHEC : Divergence significative détectée.")

if __name__ == "__main__":
    verify_radii()
