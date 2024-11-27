# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import random
from collections.abc import Mapping

from ax.core.trial import Trial
from ax.core.types import TParamValue
from pyre_extensions import none_throws


def int_from_params(
    params: Mapping[str, TParamValue], n_possibilities: int = 10
) -> int:
    """
    Get a random int between 0 and n_possibilities - 1, using parameters for the
    random seed.
    """
    seed = str(tuple(sorted(params.items())))
    return random.Random(seed).randrange(n_possibilities)


def int_from_trial(trial: Trial, n_possibilities: int = 10) -> int:
    """
    Get a random int between 0 and n_possibilities - 1, using the parameters of
    the trial's first arm for the random seed.
    """
    return int_from_params(
        params=none_throws(trial.arms)[0].parameters, n_possibilities=n_possibilities
    )
