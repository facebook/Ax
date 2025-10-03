# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Union

import numpy as np
import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis

from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment

from ax.core.parameter import ChoiceParameter, RangeParameter
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import assert_is_instance, override


class SearchSpaceAnalysis(Analysis):
    r"""
    Analysis for checking wehther the search space of the experiment should be expanded.
    It checks whether the suggested parameters land at the boundary of the search space
    and recommends expanding the search space if the proportion of the suggested
    parameters that land at the boundary is above the threshold.
    """

    def __init__(
        self, trial_index: int, boundary_proportion_threshold: float = 0.5
    ) -> None:
        r"""
        Args:
            trial_index: The index of the trial to analyze.
            boundary_proportion_threshold: The threshold on the proportion of suggested
                candidates that land on the boundary of the search space for us
                to recommend expanding the search space.

        Returns None
        """
        self.trial_index = trial_index
        self.boundary_proportion_threshold = boundary_proportion_threshold

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> HealthcheckAnalysisCard:
        if experiment is None:
            raise UserInputError("SearchSpaceAnalysis requires an Experiment.")

        status = HealthcheckStatus.PASS

        trial = experiment.trials[self.trial_index]
        arms = trial.arms
        parametrizations = [arm.parameters for arm in arms]
        boundary_proportions_df = search_space_boundary_proportions(
            search_space=experiment.search_space,
            parameterizations=parametrizations,
        )
        if np.any(
            boundary_proportions_df["Proportion of suggested values on boundary"]
            > self.boundary_proportion_threshold
        ):
            status = HealthcheckStatus.WARNING
            filtered_df = boundary_proportions_df[
                boundary_proportions_df["Proportion of suggested values on boundary"]
                > self.boundary_proportion_threshold
            ]
            filtered_df["Proportion of suggested values on boundary"] = filtered_df[
                "Proportion of suggested values on boundary"
            ].map(lambda x: f"{x * 100:.2f}%")
            filtered_df["Boundary"] = filtered_df["Boundary"].map(
                lambda x: x.replace("*", r"\*")
            )
            markdown_df = filtered_df.to_markdown(index=False)
            subtitle = (
                f"More than {self.boundary_proportion_threshold * 100:.2f}% of Ax "
                "suggested values are at the bound for the parameters and constraints "
                "below. This may indicate that the optimal values are outside of these "
                f"bounds. Consider relaxing the following bounds.\n\n{markdown_df}"
            )
        else:
            subtitle = "Search space does not need to be updated."

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title="Search space boundary check",
            subtitle=subtitle,
            df=boundary_proportions_df,
            status=status,
        )


def search_space_boundary_proportions(
    search_space: SearchSpace,
    parameterizations: list[TParameterization],
    tol: float = 1e-6,
) -> pd.DataFrame:
    r"""
    Compute the fractions of parameterizations that landed at the parameter and
    parameter constraint boundaries of the search space.

    Args:
        search_space: Search space.
        parameterizations: A list of suggested parameterizations (parameter values).
        tol: Relative tolerance for the difference between parameters and the
            boundary bounds.

    Returns:
        A dataframe containing parameters along with the fractions of parameterizations
        that landed at the parameter lower and upper limit in case of range and ordered
        choice parameters and containing parameter constraints along with the fractions
        of parameterizations that landed at the constraint boundary.
    """

    boundaries = []
    proportions = []
    bounds = []

    parameterizations = [
        parameterization
        for parameterization in parameterizations
        if search_space.check_membership(parameterization)
    ]
    num_parameterizations = len(parameterizations)

    for parameter_name, parameter in search_space.parameters.items():
        if isinstance(parameter, RangeParameter):
            lower = parameter.lower
            upper = parameter.upper
        elif isinstance(parameter, ChoiceParameter) and parameter.is_ordered:
            values = [
                assert_is_instance(v, Union[int, float]) for v in parameter.values
            ]
            lower = min(values)
            upper = max(values)
        else:
            continue
        num_lb = 0  # counts how many parameters are equal to the boundary's lower bound
        num_ub = 0  # counts how many parameters are equal to the boundary's upper bound
        for parameterization in parameterizations:
            value = parameterization[parameter_name]
            value = float(value)
            # for choice parameters, we check if the value is equal to the lower
            # or upper bound
            if isinstance(search_space.parameters[parameter_name], ChoiceParameter):
                num_lb += int(value == lower)
                num_ub += int(value == upper)
            else:
                # for range paramaters, we check if the value is within the tolerance
                if abs(value - float(lower)) < tol * (float(upper) - float(lower)):
                    num_lb += 1
                elif abs(value - float(upper)) < tol * (float(upper) - float(lower)):
                    num_ub += 1
        if num_parameterizations != 0:
            prop_lower = num_lb / float(num_parameterizations)
            prop_upper = num_ub / float(num_parameterizations)
        else:
            prop_lower = 0.0
            prop_upper = 0.0
        boundaries.extend(
            [f"{parameter_name} = {lower}", f"{parameter_name} = {upper}"]
        )
        proportions.extend([prop_lower, prop_upper])
        bounds.extend(["Lower bound", "Upper bound"])
    for pc in search_space.parameter_constraints:
        weighted_sums = [
            sum(
                float(assert_is_instance(parameterization[param], Union[int, float]))
                * weight
                for param, weight in pc.constraint_dict.items()
            )
            for parameterization in parameterizations
        ]
        if num_parameterizations != 0:
            prop = (
                np.sum(
                    [
                        abs(weighted_sum - pc.bound) < tol
                        for weighted_sum in weighted_sums
                    ]
                )
                / num_parameterizations
            )
        else:
            prop = 0.0
        boundaries.append(
            "("
            + ") + (".join(f"{v}*{k}" for k, v in sorted(pc.constraint_dict.items()))
            + f") = {pc.bound}"
        )
        proportions.append(prop)
        bounds.append("Parameter constraint")

    df = pd.DataFrame(
        {
            "Boundary": boundaries,
            "Bound": bounds,
            "Proportion of suggested values on boundary": proportions,
        }
    )
    return df
