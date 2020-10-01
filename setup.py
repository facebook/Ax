#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from setuptools import find_packages, setup


REQUIRES = [
    "botorch>=0.3.1",
    "jinja2",  # also a Plotly dep
    "pandas",
    "scipy",
    "scikit-learn",
    "plotly",
]

# pytest-cov requires pytest >= 3.6
DEV_REQUIRES = [
    "beautifulsoup4",
    "black",
    "flake8",
    "pytest>=4.6",
    "pytest-cov",
    "sphinx<3.0.0",
    "sphinx-autodoc-typehints",
    "torchvision>=0.5.0",
    "nbconvert<=5.6.1",
]

MYSQL_REQUIRES = ["SQLAlchemy>=1.1.13"]

NOTEBOOK_REQUIRES = ["jupyter"]


def setup_package() -> None:
    """Used for installing the Ax package."""

    with open("README.md", "r") as fh:
        long_description = fh.read()

    setup(
        name="ax-platform",
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
        python_requires=">=3.7",
        install_requires=REQUIRES,
        packages=find_packages(),
        package_data={
            # include all js, css, and html files in the package
            "": ["*.js", "*.css", "*.html"]
        },
        extras_require={
            "dev": DEV_REQUIRES,
            "mysql": MYSQL_REQUIRES,
            "notebook": NOTEBOOK_REQUIRES,
        },
        use_scm_version={"write_to": "ax/version.py"},
        setup_requires=["setuptools_scm"],
    )


if __name__ == "__main__":
    setup_package()
