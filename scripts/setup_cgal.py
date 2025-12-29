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
    subprocess.run(["cmake", "..", "-DCMAKE_BUILD_TYPE=Release"], cwd=build_dir, check=True)
    subprocess.run(["make", "-j"], cwd=build_dir, check=True)
    print("Build complete.")

if __name__ == "__main__":
    main()
