#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re

DOT_PLACEHOLDER = "__dot__"
SLASH_PLACEHOLDER = "__slash__"
COLON_PLACEHOLDER = "__colon__"
PIPE_PLACEHOLDER = "__pipe__"
TILDE_PLACEHOLDER = "__tilde__"
_forbidden_re: re.Pattern[str] = re.compile(r"[\;\[\'\\]")


def sanitize_name(s: str) -> str:
    """
    Converts a string with normal dots and slashes to a string with sanitized dots and
    slashes. This is temporarily necessary because SymPy symbol names must be valid
    Python identifiers, but some legacy Ax users may include dots or slashes in their
    parameter names.

    Note that we need to be careful here not to sanitize the string too much, as some
    dots are meaningful (ex. the objective "foo.bar + 0.1 * baz" should be parsed as
    "foo__dot__bar + 0.1 * baz" not "foo__dot__bar + 0__dot__1 * baz").


    This does not allow obvious attack vectors  `;`, `[`, backslashes, and quotations.
    Colons, dots, slashes, and tildes are sanitized.
    """
    if _forbidden_re.search(s) is not None:
        raise ValueError(f"Expression {s} has forbidden control characters.")
    # Replace occurances of "." and "/" when they appear after a valid Python variable
    # name and before any alphanumeric character.
    sans_dots = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\.([a-zA-Z0-9_])",
        rf"\1{DOT_PLACEHOLDER}\2",
        s,
    )
    sans_slash = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\/([a-zA-Z0-9_])",
        rf"\1{SLASH_PLACEHOLDER}\2",
        sans_dots,
    )
    sans_colon = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*):([a-zA-Z0-9_])",
        rf"\1{COLON_PLACEHOLDER}\2",
        sans_slash,
    )
    sans_pipe = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\|([a-zA-Z0-9_])",
        rf"\1{PIPE_PLACEHOLDER}\2",
        sans_colon,
    )
    # Replace tilde at the start of a variable name or after alphanumeric characters
    sans_tilde = re.sub(
        r"~([a-zA-Z_][a-zA-Z0-9_]*)",
        rf"{TILDE_PLACEHOLDER}\1",
        sans_pipe,
    )

    return sans_tilde


def unsanitize_name(s: str) -> str:
    """
    Converts a string with sanitized dots and slashes back to a string with normal dots
    and slashes. This is temporarily necessary because SymPy symbol names must be valid
    Python identifiers, but some legacy Ax users may include dots or slashes in their
    parameter names.
    """

    # Unsanitize in the reverse order of sanitization
    with_tilde = re.sub(rf"{TILDE_PLACEHOLDER}", "~", s)
    with_pipe = re.sub(rf"{PIPE_PLACEHOLDER}", "|", with_tilde)
    with_colon = re.sub(rf"{COLON_PLACEHOLDER}", ":", with_pipe)
    with_slash = re.sub(rf"{SLASH_PLACEHOLDER}", "/", with_colon)
    with_dot = re.sub(rf"{DOT_PLACEHOLDER}", ".", with_slash)

    return with_dot
