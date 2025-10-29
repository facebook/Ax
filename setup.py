# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py


def generate_thrift_files():
    """Generate Python files from thrift definitions and create __init__.py files."""
    print("Generating thrift files...")

    # Run thrift compiler on all .thrift files
    result = subprocess.run(
        "find . -name '*.thrift' -exec thrift -r --gen py --out . {} \\;",
        shell=True,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Warning: thrift compilation had issues:\n{result.stderr}")
    else:
        print("Thrift compilation completed successfully")

    # Find all ttypes.py files and create thrift_types.py in the same directory to
    # allow import from .thrift_types like we do when using FBThrift
    root_path = Path(".")
    ttypes_files = list(root_path.glob("**/ttypes.py"))

    for ttypes_file in ttypes_files:
        thrift_types_file = ttypes_file.parent / "thrift_types.py"

        with open(thrift_types_file, "w") as f:
            f.write("from .ttypes import *\n")


class CustomBuildPy(build_py):
    """Custom build_py command that generates thrift files before building."""

    def run(self):
        generate_thrift_files()
        super().run()


# Use setup() to hook in custom commands
setup(
    cmdclass={
        "build_py": CustomBuildPy,
    }
)
