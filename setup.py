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
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    language="c++",
    extra_compile_args=["-fopenmp"],
    extra_link_args=["-fopenmp"],
)

# 2. CMake Extension (CGAL Binding)
# The name "hgp_clusterer.cgal_binding" tells setuptools where to put the resulting .so
# However, CMakeLists.txt produces "cgal_binding.so".
# We set CMAKE_LIBRARY_OUTPUT_DIRECTORY to the right folder, but the filename might need check.
# pybind11_add_module uses the target name.
# So if target is "cgal_binding", it produces "cgal_binding.so".
# We want it to be inside hgp_clusterer package.
# The `extdir` calculated above ends in `build/.../hgp_clusterer/`.
# So it should work out.
cmake_ext = CMakeExtension("hgp_clusterer.cgal_binding", sourcedir=".")

setup(
    ext_modules=cythonize([cython_ext], language_level="3") + [cmake_ext],
    cmdclass={"build_ext": CMakeBuild},
)