import numpy as np
import pytest
from hgp_clusterer._cython import build_dual_graph_cython

def python_build_dual_graph_naive(simplex_indices, simplex_weights, K):
    """
    Naive Python implementation for correctness verification.
    """
    N = len(simplex_indices)
    face_map = {}
    unique_faces = []
    S_faces = []
    
    # Edges store indices of faces
    edges_u = []
    edges_v = []
    edges_w = []
    
    simplex_face_ids = np.zeros((N, K+1), dtype=int)
    
    for i in range(N):
        simp = simplex_indices[i]
        w = simplex_weights[i]
        inv_w = 1.0 / w if w > 1e-12 else 1e12
        
        # 1. Identify Faces
        for drop in range(K+1):
            # Face vertices
            face_verts = tuple(sorted([simp[t] for t in range(K+1) if t != drop]))
            
            if face_verts in face_map:
                fid = face_map[face_verts]
                S_faces[fid] += inv_w
            else:
                fid = len(unique_faces)
                face_map[face_verts] = fid
                unique_faces.append(face_verts)
                S_faces.append(inv_w)
            
            simplex_face_ids[i, drop] = fid
            
        # 2. Build Edges (0-1, 1-2, ...)
        for j in range(K):
            u = simplex_face_ids[i, j]
            v = simplex_face_ids[i, j+1]
            edges_u.append(u)
            edges_v.append(v)
            edges_w.append(w)
            
    return (np.array(unique_faces), 
            np.array(edges_u), np.array(edges_v), np.array(edges_w), 
            np.array(S_faces))


def test_dual_graph_correctness_small_K2():
    """
    Test 1: Simple manual case K=2 (Triangles -> Edges as faces)
    2 Simplices sharing an edge.
    Simplex A: [0, 1, 2] (Weight 1.0)
    Simplex B: [1, 2, 3] (Weight 2.0)
    
    Common face: {1, 2}
    """
    K = 2
    indices = np.array([
        [0, 1, 2],
        [1, 2, 3]
    ], dtype=np.int32)
    weights = np.array([1.0, 2.0], dtype=np.float32)
    
    # Run Cython
    faces, eu, ev, ew, s_faces, n_unique = build_dual_graph_cython(indices, weights, K)
    
    # Expected Faces (sorted vertices):
    # From A: {0,1}, {0,2}, {1,2}
    # From B: {1,2}, {1,3}, {2,3}
    # Unique: {0,1}, {0,2}, {1,2}, {1,3}, {2,3} -> 5 faces
    assert n_unique == 5
    assert len(faces) == 5
    
    # Check S_faces (1/r accumulation)
    # {1,2} is shared. Weight = 1/1.0 + 1/2.0 = 1.5
    # Others are single. {0,1}=1.0, {0,2}=1.0, {1,3}=0.5, {2,3}=0.5
    
    # We need to find index of {1, 2}
    # Faces are likely sorted by implicit order or generation order?
    # The new algo sorts by vertex content.
    # Sorted faces:
    # {0,1}, {0,2}, {1,2}, {1,3}, {2,3}
    
    # Verify S_faces values
    # We convert faces to list of tuples to find them
    faces_list = [tuple(f) for f in faces]
    assert (1, 2) in faces_list
    idx_shared = faces_list.index((1, 2))
    assert np.isclose(s_faces[idx_shared], 1.5)
    
    idx_single = faces_list.index((0, 1))
    assert np.isclose(s_faces[idx_single], 1.0)
    
    # Check Edges count
    # Each simplex contributes K edges (chain of K+1 faces) -> 2 * 2 = 4 edges total?
    # Logic in code: for j in range(K): link face[j] and face[j+1]
    # K=2 => j=0, 1 => 2 edges per simplex.
    assert len(eu) == 4
    assert len(ew) == 4
    

def test_dual_graph_random_K3():
    """
    Test 2: Random consistency check vs Naive Python K=3 (Tetrahedra)
    """
    K = 3
    N = 100
    n_points = 50
    np.random.seed(42)
    
    # Generate random simplices
    indices = np.random.randint(0, n_points, size=(N, K+1)).astype(np.int32)
    # Sort indices per simplex (required contract)
    indices.sort(axis=1)
    
    weights = np.random.rand(N).astype(np.float32) + 0.1
    
    # Python Baseline
    py_faces, py_eu, py_ev, py_ew, py_s = python_build_dual_graph_naive(indices, weights, K)
    
    # Cython
    cy_faces, cy_eu, cy_ev, cy_ew, cy_s, cy_n = build_dual_graph_cython(indices, weights, K)
    
    assert cy_n == len(py_faces)
    
    # Compare S_faces sum (invariant)
    assert np.isclose(np.sum(cy_s), np.sum(py_s))
    
    # Compare faces sets
    # Sort both to compare content
    py_faces_set = set(tuple(f) for f in py_faces)
    cy_faces_set = set(tuple(f) for f in cy_faces)
    assert py_faces_set == cy_faces_set
    
    # Compare max accumulated weight
    assert np.isclose(np.max(cy_s), np.max(py_s))

def test_dual_graph_high_K():
    """
    Test 3: High K stability (K=20)
    Just checks it runs and produces consistent shapes.
    """
    K = 20
    N = 50
    # Just 2 simplices identical to check full overlap
    indices = np.arange(K+1, dtype=np.int32).reshape(1, K+1)
    indices = np.vstack([indices, indices]) # Duplicate
    
    weights = np.array([1.0, 2.0], dtype=np.float32)
    
    # Cython
    cy_faces, cy_eu, cy_ev, cy_ew, cy_s, cy_n = build_dual_graph_cython(indices, weights, K)
    
    # Should have K+1 unique faces (since simplices are identical, faces match exactly)
    assert cy_n == K + 1
    
    # Weights should be 1.0 + 0.5 = 1.5 for all
    assert np.allclose(cy_s, 1.5)
