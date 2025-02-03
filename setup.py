#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup

PINNED_BOTORCH_VERSION = "0.13.0"

if os.environ.get("ALLOW_BOTORCH_LATEST"):
    # allows a more recent previously installed version of botorch to remain
    # if there is no previously installed version, installs the latest release
    botorch_req = f"botorch>={PINNED_BOTORCH_VERSION}"
else:
    botorch_req = f"botorch=={PINNED_BOTORCH_VERSION}"


REQUIRES = [
    botorch_req,
    "jinja2",  # also a Plotly dep
    "pandas",
    "scipy",
    "scikit-learn",
    "ipywidgets",
    # Needed for compatibility with ipywidgets >= 8.0.0
    "plotly>=5.12.0",
    "pyre-extensions",
    "sympy",
]

# pytest-cov requires pytest >= 3.6
DEV_REQUIRES = [
    "beautifulsoup4",
    "black==24.2.0",
    "flake8",
    "hypothesis",
    "Jinja2",
    "pyfakefs",
    "pytest>=4.6",
    "pytest-cov",
    "sphinx",
    "sphinx-autodoc-typehints",
    "sphinx_rtd_theme",
    "torchvision>=0.5.0",
    "nbconvert",
    "jupyter-client==6.1.12",
    # Replace with `tensorboard >= x.x` once tb cuts a release.
    # https://github.com/tensorflow/tensorboard/issues/6869#issuecomment-2273718763
    "numpy<2.0",
    "lxml",
    "mdformat-myst",
]

MYSQL_REQUIRES = ["SQLAlchemy==1.4.17"]

NOTEBOOK_REQUIRES = ["jupyter"]

UNITTEST_MINIMAL_REQUIRES = [
    "tensorboard",  # For tensorboard unit tests.
    "torchvision",  # For torchvision unit tests.
    "torchx",  # For torchx unit tests.
]

UNITTEST_REQUIRES = (
    DEV_REQUIRES + MYSQL_REQUIRES + NOTEBOOK_REQUIRES + UNITTEST_MINIMAL_REQUIRES
)

TUTORIAL_REQUIRES = UNITTEST_REQUIRES + [
    "ray",  # Required for building RayTune tutorial notebook.
    "tabulate",  # Required for building RayTune tutorial notebook.
    "pyarrow",  # Required for building RayTune tutorial notebook.
    "tensorboardX",  # Required for building RayTune tutorial notebook.
    "matplotlib",  # Required for building Multi-objective tutorial notebook.
    "pyro-ppl",  # Required for to call run_inference.
    "pytorch-lightning",  # For the early stopping tutorial.
    "papermill",  # For executing the tutorials.
    "submitit",  # Required for building the SubmitIt notebook.
    "mdformat",
]


def local_version(version):
    """
    Patch in a version that can be uploaded to test PyPI
    """
    return ""


def setup_package() -> None:
    """Used for installing the Ax package."""

    with open("README.md") as fh:
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
        python_requires=">=3.10",
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
            "unittest": UNITTEST_REQUIRES,
            "unittest_minimal": UNITTEST_MINIMAL_REQUIRES,
            "tutorial": TUTORIAL_REQUIRES,
        },
        use_scm_version={
            "write_to": "ax/version.py",
            "local_scheme": local_version,
        },
        setup_requires=["setuptools_scm"],
    )


if __name__ == "__main__":
    setup_package()
