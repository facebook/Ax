#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension


EXTENSIONS = [Extension("ax.utils.stats.sobol", ["ax/utils/stats/sobol.pyx"])]

REQUIRES = [
    "botorch",
    "jinja2",  # also a Plotly dep
    "pandas",
    "scipy",
    "sklearn",
    "plotly",
]

# pytest-cov requires pytest >= 3.6
DEV_REQUIRES = [
    "beautifulsoup4",
    "black",
    "flake8",
    "pytest>=3.6",
    "pytest-cov",
    "sphinx",
    "sphinx-autodoc-typehints",
]

MYSQL_REQUIRES = ["SQLAlchemy>=1.1.13"]

NOTEBOOK_REQUIRES = ["jupyter"]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="ax-platform",
    version="0.1.2",
    description="Adaptive Experimentation",
    author="Facebook, Inc.",
    license="MIT",
    url="https://github.com/facebook/Ax",
    keywords=["Experimentation", "Optimization"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.6",
    setup_requires=["cython", "numpy"],
    install_requires=REQUIRES,
    include_dirs=[numpy.get_include()],
    packages=find_packages(),
    ext_modules=cythonize(EXTENSIONS),
    package_data={
        # include all js, css, and html files in the package
        "": ["*.js", "*.css", "*.html"]
    },
    extras_require={
        "dev": DEV_REQUIRES,
        "mysql": MYSQL_REQUIRES,
        "notebook": NOTEBOOK_REQUIRES,
    },
)
