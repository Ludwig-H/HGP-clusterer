# EdgesCGALWeightedDelaunay3D & Order-K Delaunay

A high-performance C++ framework for Computational Geometry, utilizing **CGAL** and **TBB**. This project provides specialized tools for extracting Weighted Delaunay Triangulations (Regular Triangulations) and computing **Order-$k$ Delaunay** diagrams in 2D, 3D, and arbitrary dimensions.

## 🚀 Key Features

*   **Extreme Performance**:
    *   Written in **C++17** for maximum efficiency.
    *   Massively parallelized using **Intel TBB** (Threading Building Blocks).
    *   Uses **Zero-Copy / Fast I/O** via binary NumPy (`.npy`) files, avoiding slow text parsing.
    *   Optimized CGAL kernels (Inexact constructions, Exact predicates) by default.
*   **Order-$k$ Delaunay**:
    *   Implements the **Iterative Barycentric Dualization** algorithm.
    *   Computes simplices of order $k$ by lifting simplices of order $k-1$.
    *   Fully autonomous C++ pipeline: loops $k=1 … K$ internally without Python overhead.
*   **Versatility**:
    *   **2D & 3D**: Highly optimized template specializations.
    *   **ND**: Generic support via `CGAL::Regular_triangulation_d` (experimental/partial support in build).
    *   **Seamless Python Integration**: Includes a robust Python wrapper (`run_orderk.py`).

---

## 🛠️ Components

### 1. `OrderKDelaunay` (Main Tool)
The flagship tool for computing Order-$k$ Delaunay triangulations.

*   **Algorithm**:
    1.  **Step $k=1$**: Computes standard Delaunay edges.
    2.  **Step $k > 1$**: 
        *   Calculates weighted barycenters of previous simplices.
        *   Constructs a **Weighted Delaunay Triangulation** (Power Diagram dual) on these barycenters.
        *   Merges adjacent cliques to form order-$k$ candidates.
*   **Input**: `points.npy` (Float64 array of shape $N 	imes D$).
*   **Output**: `simplices.npy` (Int32 array of shape $M 	imes (K+1)$).

### 2. `EdgesCGALWeightedDelaunay3D`
A specialized utility for extracting the 1-skeleton (edges) of a 3D Regular Triangulation given points and weights.

---

## 📦 Installation & Build

### Prerequisites
*   **C++ Compiler** (GCC 9+, Clang 10+, or MSVC)
*   **CMake** (3.15+)
*   **CGAL** (5.0+)
*   **Eigen3** (Required for dD kernels)
*   **TBB** (Optional but highly recommended for parallelism)

#### Ubuntu / Debian
```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake libcgal-dev libtbb-dev libeigen3-dev libgmp-dev libmpfr-dev
```

### Compilation

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

The executables will be located in `build/`:
*   `build/OrderKDelaunay`
*   `build/EdgesCGALWeightedDelaunay3D`

---

## 🐍 Python Usage

A simplified Python wrapper `run_orderk.py` is provided to interface easily with the compiled binary.

### Quick Start

```python
import numpy as np
from run_orderk import orderk_delaunay_cpp

# 1. Generate Data
points = np.random.rand(1000, 3) # 1000 points in 3D
K = 4                            # We want order-4 simplices

# 2. Compute
# Returns a numpy array of shape (M, 5) for K=4
simplices = orderk_delaunay_cpp(points, K, verbose=True)

print(f"Found {len(simplices)} simplices of order {K}")
```

### CLI Arguments

You can also run the binary directly from the terminal:

```bash
# Usage: ./OrderKDelaunay <input.npy> <K> <output.npy> [precision] [verbose]
./build/OrderKDelaunay data.npy 3 result.npy safe 1
```

*   **`precision`**:
    *   `safe` (Default): Uses `Epick` (Exact predicates, Inexact constructions). Fast and robust for most cases.
    *   `exact`: Uses `Epeck` (Exact predicates, Exact constructions). Slower but handles degenerate cases perfectly.

---

## 🧬 Technical Details

### The Algorithm
The tool implements the connection between order-$k$ Delaunay triangulations and weighted Delaunay triangulations.
For a set of points $S$, the order-$k$ Delaunay graph can be derived by:
1.  Taking the simplices $σ$ of the order-$(k-1)$ triangulation.
2.  Computing their **centroid** $c_σ$ and a specific **weight** $w_σ$:
    $$ w_σ = \|c_σ\|^2 - \frac{1}{|\sigma|} \sum_{p \in \sigma} \|p\|^2 
$$ 
3.  Computing the **Weighted Delaunay Triangulation** of these centroids.
4.  Edges in this new triangulation correspond to pairs of $(k-1)$-simplices that can be merged to form a candidate $k$-simplex.

### Directory Structure

```
.
├── src/
│   ├── CGAL_OrderK/       # Order-K implementation
│   │   ├── main.cpp       # Main logic (Shelling/Lifting loop)
│   │   ├── kernels.hpp    # CGAL Traits & Kernel factory (2D, 3D, dD)
│   │   └── npy.hpp        # Header-only NPY I/O
│   └── EdgesCGALWeightedDelaunay3D.cpp
├── run_orderk.py          # Python wrapper script
├── CMakeLists.txt         # Build configuration
└── README.md
```

## ⚠️ Notes on High Dimensions (dD)
The `OrderKDelaunay` tool includes a templated architecture for generic dimensions (`WeightedDelaunayDD`). However, depending on the specific version of CGAL installed and the available `Epick_d` adapters, compilation of the dD module might be experimental. The 2D and 3D paths are fully specialized and stable.
