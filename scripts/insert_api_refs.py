#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import glob
import re


def list_functions(source_glob):
    """
    List all of the functions and classes defined
    """
    defined = []
    #  Iterate through each source file
    for sp in glob.glob(source_glob):
        module_name = sp[:-3]
        module_name = module_name.replace("/", ".")
        #  Parse the source file into an AST
        node = ast.parse(open(sp).read())
        #  Extract the names of all functions and classes defined in this file
        defined.extend(
            (n.name, module_name + "." + n.name)
            for n in node.body
            if (isinstance(n, ast.FunctionDef) or isinstance(n, ast.ClassDef))
        )
    return defined


def replace_backticks(source_path, docs_path):
    markdown_glob = docs_path + "/*.md"
    source_glob = source_path + "/**/*.py"
    methods = list_functions(source_glob)
    for f in glob.glob(markdown_glob):
        for n, m in methods:
            #  Match backquoted mentions of the function/class name which are
            #  not already links
            pattern = "(?<![[`])(`" + n + "`)"
            link = f"[`{n}`](/api/{m.split('.')[1]}.html#{m})"
            lines = open(f).readlines()
            for i, l in enumerate(lines):
                match = re.search(pattern, l)
                if match:
                    print(f"{f}:{i+1} s/{match.group(0)}/{link}")
                    lines[i] = re.sub(pattern, link, l)
            open(f, "w").writelines(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""In markdown docs, replace backtick-quoted names of
        objects exported from Ax with links to the API docs."""
    )
    parser.add_argument(
        "--source_path",
        metavar="source_path",
        required=True,
        help="Path to source files (e.g. 'ax/').",
    )
    parser.add_argument(
        "--docs_path", type=str, required=True, help="Path to docs (e.g. 'docs/'."
    )
    args = parser.parse_args()
    replace_backticks(args.source_path, args.docs_path)
