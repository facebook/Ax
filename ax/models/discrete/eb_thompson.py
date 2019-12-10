#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from typing import List, Tuple

import numpy as np
from ax.models.discrete.thompson import ThompsonSampler
from ax.utils.common.logger import get_logger
from ax.utils.stats.statstools import positive_part_james_stein


logger: logging.Logger = get_logger(__name__)


class EmpiricalBayesThompsonSampler(ThompsonSampler):
    """Generator for Thompson sampling using Empirical Bayes estimates.

    The generator applies positive-part James-Stein Estimator to the data
    passed in via `fit` and then performs Thompson Sampling.
    """

    def _fit_Ys_and_Yvars(
        self, Ys: List[List[float]], Yvars: List[List[float]], outcome_names: List[str]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        newYs = []
        newYvars = []
        for i, (Y, Yvar) in enumerate(zip(Ys, Yvars)):
            newY, newYvar = self._apply_shrinkage(Y, Yvar, i)
            newYs.append(newY)
            newYvars.append(newYvar)
        return newYs, newYvars

    def _apply_shrinkage(
        self, Y: List[float], Yvar: List[float], outcome: int
    ) -> Tuple[List[float], List[float]]:
        npY = np.array(Y)
        npYvar = np.array(Yvar)
        npYsem = np.sqrt(Yvar)
        try:
            npY, npYsem = positive_part_james_stein(means=npY, sems=npYsem)
        except ValueError as e:
            logger.warning(
                str(e) + f" Raw (unshrunk) estimates used for outcome: {outcome}"
            )
        Y = npY.tolist()
        npYvar = npYsem ** 2
        Yvar = npYvar.tolist()
        return Y, Yvar
