# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(kw_only=True, frozen=True)
class BenchmarkTrialMetadata:
    """
    Data pertaining to one trial evaluation.

    Args:
        Ys: A dict mapping arm names to lists of corresponding outcomes,
            where the order of the outcomes is the same as in `outcome_names`.
        Ystds: A dict mapping arm names to lists of corresponding outcome
            noise standard deviations (possibly nan if the noise level is
            unobserved), where the order of the outcomes is the same as in
            `outcome_names`.
        outcome_names: A list of metric names.
    """

    Ys: Mapping[str, Sequence[float]]
    Ystds: Mapping[str, Sequence[float]]
    outcome_names: Sequence[str]
