import numpy as np
import subprocess
import os
import sys
from pathlib import Path

def orderk_delaunay_cpp(points, k_max, output_path="output.npy", precision="safe", verbose=False):
    """
    Wraps the C++ OrderKDelaunay executable.
    """
    if points.ndim != 2:
        raise ValueError("Points must be (N, dim)")
    
    # Save points to temporary npy
    # We need to save as double (float64)
    points = points.astype(np.float64)
    input_path = "temp_input.npy"
    np.save(input_path, points)
    
    # Path to executable
    # Assuming we are in root or src, exe is in build/
    exe_path = Path("./build/OrderKDelaunay").resolve()
    if not exe_path.exists():
        # Try relative to script
        exe_path = Path(__file__).parent / "build/OrderKDelaunay"
        if not exe_path.exists():
            # Try workspace root
            exe_path = Path("/workspaces/EdgesCGALWeightedDelaunay3D/build/OrderKDelaunay")
    
    if not exe_path.exists():
        raise FileNotFoundError(f"Could not find OrderKDelaunay executable at {exe_path}")
        
    cmd = [
        str(exe_path),
        input_path,
        str(k_max),
        output_path,
        precision,
        "1" if verbose else "0"
    ]
    
    if verbose:
        print(f"Running: {' '.join(cmd)}")
        
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ tool: {e}")
        return None
    finally:
        if os.path.exists(input_path):
            os.remove(input_path)
            
    # Load result
    if os.path.exists(output_path):
        simplices = np.load(output_path)
        # It returns a list of simplices. Each row is a simplex.
        return simplices
    else:
        return None

if __name__ == "__main__":
    # Test 3D
    N = 100
    K = 3
    print(f"Generating {N} random 3D points...")
    points = np.random.rand(N, 3)
    
    output_file = "simplices.npy"
    print(f"Computing Order-{K} Delaunay...")
    
    simplices = orderk_delaunay_cpp(points, K, output_path=output_file, verbose=True)
    
    if simplices is not None:
        print(f"Success! Found {len(simplices)} simplices of order {K}.")
        print("Sample (first 5):")
        print(simplices[:5])
        
        # Verify shape
        # Order K simplices have size K+1? 
        # Wait, step K=1 -> edges (size 2).
        # Step K=2 -> triangles (size 3).
        # Step K=3 -> tetrahedra (size 4).
        # So shape should be (M, K+1).
        if simplices.shape[1] == K+1:
             print(f"Shape check passed: {simplices.shape}")
        else:
             print(f"Shape check WARNING: Expected cols {K+1}, got {simplices.shape[1]}")
    else:
        print("Failed.")
