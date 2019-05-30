#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import numpy
import subprocess
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension


EXTENSIONS = [
    Extension(
        "ax.utils.stats.sobol",
        ["ax/utils/stats/sobol.pyx"],
        include_dirs=[numpy.get_include()],
    )
]

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

def get_git_version():
    """Gets the latest tag (as a string), e.g. 0.1.2.

    Note that `git describe --tags` works as follows:
    - Finds the most recent tag that is reachable from a commit.
    - If the tag points to the commit, then only the tag is shown.
    - Otherwise, it suffixes the tag name with the number of additional commits
      on top of the tag, and the abbreviated name of the most recent commit,
      e.g. 0.1.2-9-g2118b21. If you add `--abbrev=0`, this suffix is removed,
      which is the behavior we want.
    """
    try:
        out = subprocess.check_output(
            ['git', 'describe', '--tags', '--abbrev=0'], stderr=subprocess.STDOUT
        )
        return out.strip().decode('ascii')
    except (subprocess.SubprocessError, OSError):
        return "Unknown"

def write_version_py():
    content = """#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# THIS FILE IS GENERATED FROM AX SETUP.PY

version = '%s'
"""
    f = open('ax/version.py', 'w')
    try:
        f.write(content % get_git_version())
    finally:
        f.close()


def setup_package():
    # Rewrite the version file everytime
    write_version_py()

    # Get the current version from Git
    version = get_git_version()

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="ax-platform",
        version=version,
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


if __name__ == '__main__':
    setup_package()
