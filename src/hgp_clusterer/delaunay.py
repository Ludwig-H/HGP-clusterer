import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from joblib import cpu_count

def orderk_delaunay3(
    M: np.ndarray,
    K: int,
    *,
    precision: str = "safe",
    verbose: bool = False,
    root: Path | None = None,
) -> np.ndarray:
    """
    Compute Order-K Delaunay triangulation using the optimized C++ implementation.
    """
    M = np.ascontiguousarray(M, dtype=np.float64)
    if M.ndim != 2:
        raise ValueError("M must be 2D")
    if K < 1:
        raise ValueError("K must be >= 1")
    n, d = M.shape
    if n < 2:
        return []

    # Locate binary
    root_dir = root or os.environ.get("CGALDELAUNAY_ROOT")
    if root_dir is None:
        # Assuming we are in src/hgp_clusterer/, go up 2 levels
        root_dir = Path(__file__).resolve().parents[2] / "CGALDelaunay"
    else:
        root_dir = Path(root_dir)
    
    binary = root_dir / "orderk_delaunay_cpp" / "build" / "orderk_delaunay_cpp"
    if not binary.exists():
        raise FileNotFoundError(f"CGAL binary not found: {binary}. Please build it.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        input_file = tmp_path / "input.npy"
        output_file = tmp_path / "output.npy"
        
        np.save(input_file, M)
        
        # Binary Usage: input.npy K output.npy [precision] [verbose]
        cmd = [
            str(binary),
            str(input_file),
            str(K),
            str(output_file),
            precision,
            "1" if verbose else "0"
        ]
        
        env = os.environ.copy()
        if "CGAL_NTHREADS" not in env:
             env["CGAL_NTHREADS"] = str(max(1, cpu_count()))
        
        subprocess.run(cmd, check=True, env=env)
        
        if not output_file.exists():
            return []
            
        try:
            result = np.load(output_file)
        except (ValueError, EOFError):
            return []
        
    if result.size == 0:
        return np.empty((0, K + 1), dtype=np.int64)
        
    return result # Returns np.ndarray (M, K+1) directly


