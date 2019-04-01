#!/usr/bin/env python3

import logging
from typing import List, Optional, Tuple

import numpy as np
from ax.models.discrete.eb_thompson import EmpiricalBayesThompsonSampler
from ax.utils.common.logger import get_logger
from ax.utils.stats.statstools import (
    ancillary_james_stein,
    relativize_rates_against_mean,
)


logger: logging.Logger = get_logger(__name__)


class AncillaryEBThompsonSampler(EmpiricalBayesThompsonSampler):
    """Generator for Thompson sampling using Ancillary Empirical Bayes estimates.

    The generator applies the Ancillary James-Stein Estimator to the data
    passed in via `fit` and then performs Thompson Sampling.
    """

    def __init__(
        self,
        primary_outcome: str,
        secondary_outcome: str,
        num_samples: int = 10000,
        min_weight: Optional[float] = None,
        uniform_weights: bool = False,
    ) -> None:
        self.primary_outcome = primary_outcome
        self.secondary_outcome = secondary_outcome
        super(AncillaryEBThompsonSampler, self).__init__(
            num_samples=num_samples,
            min_weight=min_weight,
            uniform_weights=uniform_weights,
        )

    def _fit_Ys_and_Yvars(
        self, Ys: List[List[float]], Yvars: List[List[float]], outcome_names: List[str]
    ) -> Tuple[List[List[float]], List[List[float]]]:
        primary_outcome_index = outcome_names.index(self.primary_outcome)
        secondary_outcome_index = outcome_names.index(self.secondary_outcome)

        self._check_input_validity(
            Ys, Yvars, primary_outcome_index, secondary_outcome_index
        )
        newYs = []
        newYvars = []
        for i, (Y, Yvar) in enumerate(zip(Ys, Yvars)):
            if i == primary_outcome_index:
                Y = np.array(Y)
                Yvar = np.array(Yvar)
                Y2 = np.array(Ys[secondary_outcome_index])
                Y2var = np.array(Yvars[secondary_outcome_index])
                imputed_var_primary = Y * (1 - Y)
                imputed_var_primary = np.where(
                    imputed_var_primary == 0, 0.25, imputed_var_primary
                )
                imputed_var_secondary = Y2 * (1 - Y2)
                imputed_var_secondary = np.where(
                    imputed_var_secondary == 0, 0.25, imputed_var_secondary
                )
                N = np.round(np.power(Yvar / imputed_var_primary, -1))
                N2 = np.round(np.power(Y2var / imputed_var_secondary, -1))
                newY, newYvar = self._apply_ancillary_eb(Y, N, Y2, N2)
            else:
                newY, newYvar = self._apply_shrinkage(Y, Yvar, i)
            newYs.append(newY)
            newYvars.append(newYvar)
        return newYs, newYvars

    def _check_input_validity(
        self,
        Ys: List[List[float]],
        Yvars: List[List[float]],
        primary_outcome_index: int,
        secondary_outcome_index: int,
    ) -> None:
        if primary_outcome_index == secondary_outcome_index:
            raise ValueError("Primary and secondary outcomes must differ.")

        Y1 = Ys[primary_outcome_index]
        Y2 = Ys[secondary_outcome_index]
        Y1v = Yvars[primary_outcome_index]
        Y2v = Yvars[secondary_outcome_index]

        if any(v > 0.25 for v in Y1v) or any(v > 0.25 for v in Y2v):
            raise ValueError(
                "Variances cannot be greater than 0.25 for binary outcome."
            )

        if any(m < 0 or m > 1 for m in Y1) or any(m < 0 or m > 1 for m in Y2):
            raise ValueError("Means of binary outcome must lie in [0, 1].")

    def _apply_ancillary_eb(
        self, Y1: np.ndarray, N1: np.ndarray, Y2: np.ndarray, N2: np.ndarray
    ) -> Tuple[List[float], List[float]]:
        overall_p = np.average(Y1, weights=N1)
        r_unbiased, s_unbiased = relativize_rates_against_mean(Y1, N1)
        r_biased, s_biased = relativize_rates_against_mean(Y2, N2)
        eb_est, eb_sem = ancillary_james_stein(
            r_unbiased, s_unbiased, r_biased, s_biased, N1
        )
        # We rescale these estimates back to the scale of the original variable
        # so that all of our usual tools look more or less like usual.
        return (
            ((1 + eb_est) * overall_p).tolist(),
            np.power(eb_sem * overall_p, 2).tolist(),
        )
