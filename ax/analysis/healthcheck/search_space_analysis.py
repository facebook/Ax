# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from typing import Union

import numpy as np
import pandas as pd

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface

from ax.core.parameter import ChoiceParameter, Parameter, RangeParameter
from ax.core.parameter_constraint import ParameterConstraint
from ax.core.search_space import SearchSpace
from ax.core.types import TParameterization
from ax.exceptions.core import UserInputError
from pyre_extensions import assert_is_instance


class SearchSpaceAnalysis(HealthcheckAnalysis):
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

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> HealthcheckAnalysisCard:
        r"""
        Args:
            experiment: Ax experiment.
            generation_strategy: Ax generation strategy.

        Returns:
            A HealthcheckAnalysisCard object with the information on the parameters
            and parameter constraints whose boundaries are recommended to be expanded.
        """

        if experiment is None:
            raise UserInputError("SearchSpaceAnalysis requires an Experiment.")

        status = HealthcheckStatus.PASS
        subtitle = "Search space does not need to be updated."
        title_status = "Success"
        level = AnalysisCardLevel.LOW
        df = pd.DataFrame({"status": [status]})

        trial = experiment.trials[self.trial_index]
        arms = trial.arms
        parametrizations = [arm.parameters for arm in arms]
        boundary_proportions_df = search_space_boundary_proportions(
            search_space=experiment.search_space,
            parametrizations=parametrizations,
        )
        if np.any(
            boundary_proportions_df["proportion"] > self.boundary_proportion_threshold
        ):
            msg = boundary_proportions_message(
                boundary_proportions_df=boundary_proportions_df,
                boundary_proportion_threshold=self.boundary_proportion_threshold,
            )
            status = HealthcheckStatus.WARNING
            subtitle = msg
            title_status = "Warning"
            level = AnalysisCardLevel.LOW
            df = boundary_proportions_df[["boundary", "proportion", "bound"]]
            df["status"] = status

        return HealthcheckAnalysisCard(
            name="SearchSpaceAnalysis",
            title=f"Ax Search Space Analysis {title_status}",
            blob=json.dumps({"status": status}),
            subtitle=subtitle,
            df=df,
            level=level,
            attributes={"trial_index": self.trial_index},
        )


def search_space_boundary_proportions(
    search_space: SearchSpace,
    parametrizations: list[TParameterization],
    tol: float = 1e-6,
) -> pd.DataFrame:
    r"""
    Compute the fractions of parametrizations that landed at the parameter and
    parameter constraint boundaies of the search space.

    Args:
        search_space: Search space.
        parametrizations: A list of suggested parametrizations (parameter values).
        tol: Relative tolerance for the difference between parameters and the
            boundary bounds.

    Returns:
        A dataframe containing parameters along with the fractions of parametrizations
        that landed at the parameter lower and upper limit in case of range and ordered
        choice parameters and containing parameter constraints along with the fractions
        of parametrizations that landed at the constraint boundary.
    """

    parameters_and_constraints = []
    boundaries = []
    proportions = []
    bounds = []

    num_parametrizations = len(parametrizations)

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
        for parametrization in parametrizations:
            value = parametrization[parameter_name]
            if value is None:
                continue
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
        prop_lower = num_lb / float(num_parametrizations)
        prop_upper = num_ub / float(num_parametrizations)
        parameters_and_constraints.extend([parameter] * 2)
        boundaries.extend(
            [f"{parameter_name} = {lower}", f"{parameter_name} = {upper}"]
        )
        proportions.extend([prop_lower, prop_upper])
        bounds.extend(["lower", "upper"])

    for pc in search_space.parameter_constraints:
        weighted_sums = [
            sum(
                float(assert_is_instance(parametrization[param], Union[int, float]))
                * weight
                for param, weight in pc.constraint_dict.items()
            )
            for parametrization in parametrizations
        ]
        prop = (
            np.sum(
                [abs(weighted_sum - pc.bound) < tol for weighted_sum in weighted_sums]
            )
            / num_parametrizations
        )
        boundaries.append(
            " + ".join(f"{v}*{k}" for k, v in sorted(pc.constraint_dict.items()))
            + f" = {pc.bound}"
        )
        proportions.append(prop)
        parameters_and_constraints.append(pc)
        bounds.append("upper")

    df = pd.DataFrame(
        {
            "parameter_or_constraint": parameters_and_constraints,
            "boundary": boundaries,
            "proportion": proportions,
            "bound": bounds,
        }
    )
    return df


def boundary_proportions_message(
    boundary_proportions_df: pd.DataFrame,
    boundary_proportion_threshold: float = 0.5,
) -> str:
    r"""
    Construct a message explaning what parameter or parameter constraints bounds
    to change based on the proportions of the parametrizations that landed at
    the search spaces boundaries.
    A proportion should be above the theshold in order to recommend expanding
    the search space along the corresponding parameter or parameter constraint.

    Args:
        boundary_proportions_df: A dataframe with the following columns
            * parameter_or_constraint: the parameter or constraint object
                containing this row's search space boundary.
            * boundary: a string representation of the function defining this boundary.
            * proportion: the proportion of provided parameterizations within
                ``tol`` of this boundary.
            * bound: whether this is an upper or lower bound.
        A dataframe containing parameters and parameter
            constraints along with the proportions of the parametrizations that
            landed at the lower or upper bounds of the parameters or at the
            constraints boundary.
        boundary_proportion_threshold: The minimal proportion of suggested
            parametrizations that land at the boundary of the search space for
            us to recommend expanding the search space.

    Returns:
        A string explaning what parameter or parameter constraints bounds to change
        in order to expand the search space.
    """

    msg = ""
    for _, row in boundary_proportions_df.iterrows():
        if isinstance(row["parameter_or_constraint"], Parameter):
            parameter = row["parameter_or_constraint"]
            bound = row["bound"]
            prop = row["proportion"]
            if bound == "lower" and prop >= boundary_proportion_threshold:
                msg += (
                    f"\n - Parameter {parameter.name} values are at their lower bound "
                    f"in {prop * 100:.2f}% of all suggested parameters, which exceeds "
                    f"the threshold of {boundary_proportion_threshold * 100:.2f}%. "
                    "Consider decreasing this lower bound of the search space and "
                    "re-generating the candidates inside the expanded search space. "
                )

            if bound == "upper" and prop >= boundary_proportion_threshold:
                msg += (
                    f"\n - Parameter {parameter.name} values are at its upper bound "
                    f"in {prop * 100:.2f}% of all suggested parameters, which exceeds "
                    f"the threshold of {boundary_proportion_threshold * 100:.2f}%. "
                    "Consider increasing this upper bound of the search space and "
                    "re-generating the candidates inside the expanded search space. "
                )
        elif isinstance(row["parameter_or_constraint"], ParameterConstraint):
            pc = row["parameter_or_constraint"]
            prop = row["proportion"]
            if prop >= boundary_proportion_threshold:
                msg += (
                    f"\n - Parameter constraint {pc} is binding for {prop * 100:.2f}% "
                    " of all suggested parameters, which exceeds the threshold of "
                    f"{boundary_proportion_threshold * 100:.2f}%. "
                    "Consider increasing this constraint bound and re-generating the "
                    "candidates inside the expanded search space. "
                )

    return msg
