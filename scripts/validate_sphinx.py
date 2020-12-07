#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import pkgutil
import re
from typing import Set


# Paths are relative to top-level Ax directory (which is passed into fxn below)
SPHINX_RST_PATH = os.path.join("sphinx", "source")
AX_LIBRARY_PATH = "ax"

# Regex for automodule directive used in Sphinx docs
AUTOMODULE_REGEX = re.compile(r"\.\. automodule:: ([\.\w]*)")

# Modules to exclude from validation
EXCLUDE_MODULES = {
    "ax.utils.testing.doctest",
    "ax.utils.testing.fully_annotated",
    "ax.utils.testing.manifest",
    "ax.utils.testing.pyre_strict",
}


def parse_rst(rst_filename: str) -> Set[str]:
    """Extract automodule directives from rst."""
    ret = set()
    with open(rst_filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            name = AUTOMODULE_REGEX.findall(line)
            if name:
                ret.add(name[0])
    return ret


def validate_complete_sphinx(path_to_ax: str) -> None:
    """Validate that Sphinx-based API documentation is complete.

    * Every top-level module (e.g., core, models, etc.) should have corresponding rst
      file.
    * Every single non-package (i.e. py file) module should be included in rst file with
      `automodule::` directive. Sphinx will then automatically include all members from
       the module in the documentation.

    Note: this function does not validate any documentation for the 'ax' module.

    Args:
      path_to_ax: the path to the top-level ax directory (directory that includes ax
        library, sphinx, website, etc.).

    """
    # Load top-level modules used in Ax (e.g., core, models; exclude 'fb' and 'version')
    modules = {
        modname
        for importer, modname, ispkg in pkgutil.walk_packages(
            path=[AX_LIBRARY_PATH], onerror=lambda x: None
        )
        if modname not in {"fb", "version"}
    }

    # Load all rst files (these contain the documentation for Sphinx)
    rstpath = os.path.join(path_to_ax, SPHINX_RST_PATH)
    rsts = {f.replace(".rst", "") for f in os.listdir(rstpath) if f.endswith(".rst")}

    # Verify that all top-level modules have a corresponding rst
    assert len(modules.difference(rsts)) == 0, "Not all modules have corresponding rst."

    # Track all modules that are not in docs (so can print all)
    modules_not_in_docs = []

    # Iterate over top-level modules
    for module in modules.intersection(rsts):
        # Parse rst & extract all modules use automodule directive
        modules_in_rst = parse_rst(os.path.join(rstpath, module + ".rst"))

        # Extract all non-package modules
        for _importer, modname, ispkg in pkgutil.walk_packages(
            path=[os.path.join(AX_LIBRARY_PATH, module)],  # ax.__path__[0], module),
            prefix="ax." + module + ".",
            onerror=lambda x: None,
        ):
            if (
                not ispkg
                and ".tests" not in modname
                and modname not in modules_in_rst.union(EXCLUDE_MODULES)
            ):
                modules_not_in_docs.append(modname)

    assert len(modules_not_in_docs) == 0, "Not all modules are documented: {}".format(
        modules_not_in_docs
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate that Sphinx documentation is complete."
    )
    parser.add_argument(
        "-p",
        "--path",
        metavar="path",
        required=True,
        help="Path to the top-level ax directory.",
    )
    args = parser.parse_args()
    validate_complete_sphinx(args.path)
