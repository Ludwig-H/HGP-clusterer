from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

import pybind11

def get_eigen_include():
    paths = [
        "/usr/include/eigen3",
        "/usr/local/include/eigen3",
        "/opt/homebrew/include/eigen3",
        "/opt/local/include/eigen3"
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    # Fallback, let compiler find it if it's in standard path
    return ""

eigen_path = get_eigen_include()
include_dirs = [np.get_include()]
if eigen_path:
    include_dirs.append(eigen_path)

# A CMakeExtension needs a sourcedir
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)

class CMakeBuild(build_ext):
    def build_extension(self, ext: Extension) -> None:
        # If it's not a CMakeExtension, use the default build_ext logic (for Cython)
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        # Check for CMake
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " + ext.name)

        # Output directory for the extension
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))

        # CMake configuration args
        # cfg: Debug or Release
        cfg = "Debug" if self.debug else "Release"
        
        # Generator: Ninja is faster if available
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
        ]

        if "CMAKE_PREFIX_PATH" in os.environ:
            # CMake uses semicolon as list separator
            prefix_path = os.environ['CMAKE_PREFIX_PATH'].replace(os.pathsep, ";")
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={prefix_path}")
            
            # Heuristic: If specific CGAL path is in prefix, set CGAL_DIR explicitly
            for path in os.environ['CMAKE_PREFIX_PATH'].split(os.pathsep):
                if path.endswith("CGAL") and "cmake" in path:
                    cmake_args.append(f"-DCGAL_DIR={path}")
                    break
        
        # Multi-config generators (like Visual Studio) need simplified config
        build_args = ["--config", cfg]

        # Handle parallel builds
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
             if hasattr(self, "parallel") and self.parallel:
                 build_args += [f"-j{self.parallel}"]

        # Ensure output directory exists
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # Run CMake Configure
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        
        # Run CMake Build
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )

# 1. Cython Extension
cython_ext = Extension(
    "hgp_clusterer._cython",
    sources=[str(Path("src") / "hgp_clusterer" / "_cython.pyx")],
    include_dirs=include_dirs,
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++",
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

# 2. CMake Extension (CGAL/Geogram Binding)
# The name "hgp_clusterer.geometry_binding" tells setuptools where to put the resulting .so
# However, CMakeLists.txt produces "geometry_binding.so".
# We set CMAKE_LIBRARY_OUTPUT_DIRECTORY to the right folder, but the filename might need check.
# pybind11_add_module uses the target name.
# So if target is "geometry_binding", it produces "geometry_binding.so".
# We want it to be inside hgp_clusterer package.
# The `extdir` calculated above ends in `build/.../hgp_clusterer/`.
# So it should work out.
cmake_ext = CMakeExtension("hgp_clusterer.geometry_binding", sourcedir=".")

setup(
    ext_modules=cythonize([cython_ext], language_level="3") + [cmake_ext],
    cmdclass={"build_ext": CMakeBuild},
)