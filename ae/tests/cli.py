#!/usr/bin/env python3

"""Main module for the lazarus test runner

This is just a wrapper around testslides main module that injects extra tests based on
a information written out by the `ae_unittest` buck macro.
"""
import sys

import __test_modules__
from __ae__test_options__ import TEST_OPTIONS
from libfb.py.testslide.cli import Cli


def main():
    import_module_names = __test_modules__.TEST_MODULES.copy()
    root = "ae.lazarus.ae.tests"
    if TEST_OPTIONS["doctest"]:
        import_module_names.append(f"{root}.doctest")
    if TEST_OPTIONS["pyre_strict"]:
        import_module_names.append(f"{root}.test_pyre_strict")
    if TEST_OPTIONS["fully_annotated"]:
        import_module_names.append(f"{root}.test_fully_annotated")
    import_module_names.append(f"{root}.test_unittest_conventions")
    Cli(sys.argv[1:], import_module_names=import_module_names).run()


if __name__ == "__main__":
    main()
