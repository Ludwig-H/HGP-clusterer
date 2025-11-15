#!/usr/bin/env python3
from __future__ import annotations

import subprocess
from pathlib import Path

REPOS = [
    "https://github.com/Ludwig-H/EdgesCGALWeightedDelaunay3D.git",
    "https://github.com/Ludwig-H/EdgesCGALWeightedDelaunay2D.git",
    "https://github.com/Ludwig-H/EdgesCGALWeightedDelaunayND.git",
    "https://github.com/Ludwig-H/EdgesCGALDelaunay3D.git",
    "https://github.com/Ludwig-H/EdgesCGALDelaunay2D.git",
    "https://github.com/Ludwig-H/EdgesCGALDelaunayND.git",
]


def main() -> None:
    root = Path(__file__).resolve().parents[1] / "CGALDelaunay"
    root.mkdir(exist_ok=True)
    for url in REPOS:
        name = url.rstrip("/").split("/")[-1].removesuffix(".git")
        dest = root / name
        if dest.exists():
            print(f"[skip] {name} already exists")
            continue
        print(f"[clone] {url} -> {dest}")
        subprocess.run(["git", "clone", "--depth", "1", url, str(dest)], check=True)


if __name__ == "__main__":
    main()
