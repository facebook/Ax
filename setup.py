#!/usr/bin/env python3

import numpy
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension


EXTENSIONS = [Extension("ae.utils.stats.sobol", ["ae/utils/stats/sobol.pyx"])]

REQUIRES = [
    "botorch",
    "GPy==1.9.6",  # will be removed prior to release
    "jinja2",  # also a Plotly dep
    "matplotlib",  # will be removed prior to release (needed for GPy)
    "pandas",
    "scipy",
    "simplejson",
    "sklearn",
    "plotly==2.4.1",
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


setup(
    name="ae",
    version="pre-alpha",
    description="Adaptive Experimentation",
    author="Facebook, Inc.",
    license="MIT",
    url="https://github.com/facebook/Adaptive-Experiment",
    keywords=["Experimentation", "Optimization"],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
    ],
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
