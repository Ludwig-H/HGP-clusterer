import numpy as np
import subprocess
import os
import sys

def test_dimension(dim, n_points=20, k=2):
    print(f"\n--- Testing Dimension {dim} ---")
    
    # Generate random points
    points = np.random.rand(n_points, dim).astype(np.float64)
    
    # Save to temporary npy
    input_file = f"temp_input_{dim}d.npy"
    output_file = f"temp_output_{dim}d.npy"
    
    np.save(input_file, points)
    
    # Path to binary (built in previous step)
    binary_path = "./CGALDelaunay/orderk_delaunay_cpp/build/orderk_delaunay_cpp"
    
    if not os.path.exists(binary_path):
        print("Error: Binary not found at", binary_path)
        return

    # Command: binary input K output precision verbose
    cmd = [binary_path, input_file, str(k), output_file, "safe", "1"]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
            
        if result.returncode == 0:
            if os.path.exists(output_file):
                simplices = np.load(output_file)
                print(f"Success! Output shape: {simplices.shape}")
                print(f"First 5 simplices:\n{simplices[:5]}")
            else:
                print("Error: Output file not created.")
        else:
            print("Error: C++ executable returned non-zero exit code.")
            
    except Exception as e:
        print(f"Exception: {e}")
        
    # Cleanup
    if os.path.exists(input_file):
        os.remove(input_file)
    if os.path.exists(output_file):
        os.remove(output_file)

if __name__ == "__main__":
    # Test D=1 (Often degenerate for Delaunay 2D/3D kernels, might fallback to dD)
    test_dimension(1)
    
    # Test D=2 (Standard Optimized 2D)
    test_dimension(2)
    
    # Test D=3 (Standard Optimized 3D)
    test_dimension(3)
    
    # Test D=4 (Requires dD, currently stubbed)
    test_dimension(4)
