from __future__ import annotations

from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup


EXTENSIONS = [
    Extension(
        "hypergraphpercol._cython",
        sources=[str(Path("src") / "hgp_clusterer" / "_cython.pyx")],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]


setup(ext_modules=cythonize(EXTENSIONS, language_level="3"))
