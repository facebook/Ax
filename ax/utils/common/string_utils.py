#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import re

import sympy

DOT_PLACEHOLDER = "__dot__"
SLASH_PLACEHOLDER = "__slash__"
COLON_PLACEHOLDER = "__colon__"
PIPE_PLACEHOLDER = "__pipe__"
TILDE_PLACEHOLDER = "__tilde__"
SPACE_PLACEHOLDER = "__space__"
HYPHEN_PLACEHOLDER = "__hyphen__"
_forbidden_re: re.Pattern[str] = re.compile(r"[\;\[\'\\]")

SYMPY_GLOBALS: set[str] = set(dir(sympy))

# Allow some globals, like oo (infinity) to be used in expressions.
ALLOWED_GLOBALS: set[str] = {"oo"}


def _check_sympy_conflicts(expression: str) -> None:
    """
    Check if the expression contains identifiers that conflict with sympy's global dict.

    Raises ValueError if a conflict is detected.
    """
    # Extract all Python identifiers from the expression
    # This regex matches valid Python identifiers
    identifier_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_()]*)\b"
    identifiers = set(re.findall(identifier_pattern, expression))

    # Check for conflicts
    conflicts = (identifiers & SYMPY_GLOBALS) - ALLOWED_GLOBALS

    if conflicts:
        conflicts_list = ", ".join(sorted(conflicts))
        raise ValueError(
            f"Expression '{expression}' contains identifiers that conflict with "
            f"sympy's built-in names: {conflicts_list}. "
            f"Please rename these variables to avoid conflicts."
        )


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
    # Replace spaces between word characters so metric names with spaces
    # (e.g. "CIFAR10 Test Accuracy") become valid Python identifiers.
    # Spaces around operators (e.g. "m1 + m2") are not affected because
    # operators like +, -, *, are not word characters.
    sans_space = re.sub(r"(?<=\w)\s+(?=\w)", SPACE_PLACEHOLDER, s)
    # Replace occurrences of "." and "/" when they appear after a valid Python variable
    # name and before any alphanumeric character.  We use a lookahead (?=...)
    # for the trailing character so it is NOT consumed by the match; this
    # allows chained separators (e.g. "a:b:c:d") to be fully sanitized in a
    # single pass.
    sans_dots = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\.(?=[a-zA-Z0-9_])",
        rf"\1{DOT_PLACEHOLDER}",
        sans_space,
    )
    sans_slash = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\/(?=[a-zA-Z0-9_])",
        rf"\1{SLASH_PLACEHOLDER}",
        sans_dots,
    )
    sans_colon = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*):(?=[a-zA-Z0-9_])",
        rf"\1{COLON_PLACEHOLDER}",
        sans_slash,
    )
    sans_pipe = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]*)\|(?=[a-zA-Z0-9_])",
        rf"\1{PIPE_PLACEHOLDER}",
        sans_colon,
    )
    # Replace tilde at the start of a variable name or after alphanumeric characters
    sans_tilde = re.sub(
        r"~([a-zA-Z_][a-zA-Z0-9_]*)",
        rf"{TILDE_PLACEHOLDER}\1",
        sans_pipe,
    )
    # Replace hyphens when they appear between identifier characters
    # (e.g. "120s-300s" in metric names like "metric:120s-300s").
    # We require at least 2 characters in the preceding identifier to avoid
    # false positives on scientific notation (e.g. "2.3E-4").
    sans_hyphen = re.sub(
        r"([a-zA-Z_][a-zA-Z0-9_]+)-(?=[a-zA-Z0-9_])",
        rf"\1{HYPHEN_PLACEHOLDER}",
        sans_tilde,
    )
    # Check for conflicts with sympy's global dictionary
    _check_sympy_conflicts(sans_hyphen)

    return sans_hyphen


def unsanitize_name(s: str) -> str:
    """
    Converts a string with sanitized dots and slashes back to a string with normal dots
    and slashes. This is temporarily necessary because SymPy symbol names must be valid
    Python identifiers, but some legacy Ax users may include dots or slashes in their
    parameter names.
    """

    # Unsanitize in the reverse order of sanitization
    with_hyphen = re.sub(rf"{HYPHEN_PLACEHOLDER}", "-", s)
    with_tilde = re.sub(rf"{TILDE_PLACEHOLDER}", "~", with_hyphen)
    with_pipe = re.sub(rf"{PIPE_PLACEHOLDER}", "|", with_tilde)
    with_colon = re.sub(rf"{COLON_PLACEHOLDER}", ":", with_pipe)
    with_slash = re.sub(rf"{SLASH_PLACEHOLDER}", "/", with_colon)
    with_dot = re.sub(rf"{DOT_PLACEHOLDER}", ".", with_slash)
    with_space = re.sub(rf"{SPACE_PLACEHOLDER}", " ", with_dot)

    return with_space
