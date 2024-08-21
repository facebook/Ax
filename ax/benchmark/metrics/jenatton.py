# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Optional

from ax.utils.common.typeutils import not_none


def jenatton_test_function(
    x1: Optional[int] = None,
    x2: Optional[int] = None,
    x3: Optional[int] = None,
    x4: Optional[float] = None,
    x5: Optional[float] = None,
    x6: Optional[float] = None,
    x7: Optional[float] = None,
    r8: Optional[float] = None,
    r9: Optional[float] = None,
) -> float:
    """Jenatton test function for hierarchical search spaces.

    This function is taken from:

    R. Jenatton, C. Archambeau, J. González, and M. Seeger. Bayesian
    optimization with tree-structured dependencies. ICML 2017.
    """
    if x1 == 0:
        if x2 == 0:
            return not_none(x4) ** 2 + 0.1 + not_none(r8)
        else:
            return not_none(x5) ** 2 + 0.2 + not_none(r8)
    else:
        if x3 == 0:
            return not_none(x6) ** 2 + 0.3 + not_none(r9)
        else:
            return not_none(x7) ** 2 + 0.4 + not_none(r9)
