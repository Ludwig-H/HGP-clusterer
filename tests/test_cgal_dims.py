import numpy as np
import subprocess
import os
import sys
import pytest

@pytest.mark.parametrize("dim", [1, 2, 3])
def test_dimension(dim):
    n_points = 20
    k = 2
    print(f"\n--- Testing Dimension {dim} ---")
    
    # Generate random points
    points = np.random.rand(n_points, dim).astype(np.float64)
    
    # Save to temporary npy
    input_file = f"temp_input_{dim}d.npy"
    output_file = f"temp_output_{dim}d.npy"
    
    np.save(input_file, points)
    
    # Path to binary (built in previous step)
    # Adjust path relative to test execution
    binary_path = "./CGALDelaunay/orderk_delaunay_cpp/build/orderk_delaunay_cpp"
    if not os.path.exists(binary_path):
        # Try absolute path based on workspace
        binary_path = "/workspaces/HGP-clusterer/CGALDelaunay/orderk_delaunay_cpp/build/orderk_delaunay_cpp"
    
    if not os.path.exists(binary_path):
        pytest.skip(f"Binary not found at {binary_path}")
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
                pytest.fail("Error: Output file not created.")
        else:
            pytest.fail("Error: C++ executable returned non-zero exit code.")
            
    except Exception as e:
        pytest.fail(f"Exception: {e}")
    finally:
        # Cleanup
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

if __name__ == "__main__":
    pytest.main([__file__])
