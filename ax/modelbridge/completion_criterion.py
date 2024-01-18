# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from logging import Logger

from ax.modelbridge.transition_criterion import (
    MinimumPreferenceOccurances,
    TransitionCriterion,
)
from ax.utils.common.logger import get_logger

logger: Logger = get_logger(__name__)


class CompletionCriterion(TransitionCriterion):
    """
    Deprecated class that has been replaced by `TransitionCriterion`, and will be
    fully reaped in a future release.
    """

    logger.warning(
        "CompletionCriterion is deprecated, please use TransitionCriterion instead."
    )
    pass


class MinimumPreferenceOccurances(MinimumPreferenceOccurances):
    """
    Deprecated child class that has been replaced by `MinimumPreferenceOccurances`
    in `TransitionCriterion`, and will be fully reaped in a future release.
    """

    logger.warning(
        "CompletionCriterion, which MinimumPreferenceOccurance inherits from, is"
        " deprecated. Please use TransitionCriterion instead."
    )
    pass
