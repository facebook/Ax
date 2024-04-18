# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import re


def is_valid_name(name: str) -> bool:
    """
    Check if a name is valid (ex. for a Metric's or Parameter's name field).

    Naming conventions are very similar to those of Python variables, however colons
    are allowed as well:
        - Names must be non-empty
        - Names must start with a letter or underscore
        - After the first character, names can contain letters, numbers, underscores
            and colons
    """
    return re.fullmatch(r"[_a-zA-Z][_a-zA-Z0-9:]*", name) is not None
