# CGAL helper binaries

This directory should contain the compiled CGAL helper executables:

- `EdgesCGALWeightedDelaunay2D`
- `EdgesCGALWeightedDelaunay3D`
- `EdgesCGALWeightedDelaunayND`
- `EdgesCGALDelaunay2D`
- `EdgesCGALDelaunay3D`
- `EdgesCGALDelaunayND`

Run `./scripts/setup_cgal.py` to clone the corresponding repositories here. After cloning, build each project with CMake following their respective README instructions. HypergraphPercol expects the executables to live in `build/` inside each project folder.
