#!/usr/bin/env python3
import subprocess
import os
from pathlib import Path

def main():
    root = Path(__file__).resolve().parents[1] / "CGALDelaunay" / "orderk_delaunay_cpp"
    if not root.exists():
        print(f"Error: {root} does not exist.")
        return

    build_dir = root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Building C++ tool in {build_dir}...")
    cmd = ["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"]
    
    if "CMAKE_PREFIX_PATH" in os.environ:
        cmd.append(f"-DCMAKE_PREFIX_PATH={os.environ['CMAKE_PREFIX_PATH'].replace(os.pathsep, ';')}")
        for path in os.environ['CMAKE_PREFIX_PATH'].split(os.pathsep):
            if path.endswith("CGAL") and "cmake" in path:
                cmd.append(f"-DCGAL_DIR={path}")
                break

    subprocess.run(cmd, cwd=build_dir, check=True)
    subprocess.run(["make", "-j1"], cwd=build_dir, check=True)
    print("Build complete.")

if __name__ == "__main__":
    main()
