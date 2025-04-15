# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import Sequence

import pandas as pd

from ax.analysis.analysis import AnalysisCardCategory, AnalysisCardLevel

from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.plotly.arm_effects.utils import get_predictions_by_arm
from ax.analysis.plotly.utils import is_predictive
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.modelbridge.base import Adapter
from ax.modelbridge.transforms.derelativize import Derelativize
from pyre_extensions import assert_is_instance, override


class ConstraintsFeasibilityAnalysis(HealthcheckAnalysis):
    """
    Analysis for checking the feasibility of the constraints for the experiment.
    A constraint is considered feasible if the probability of constraints violation
    is below the threshold for at least one arm.
    """

    def __init__(self, prob_threshold: float = 0.95) -> None:
        r"""
        Args:
            prob_threhshold: Threshold for the probability of constraint violation.

        """
        self.prob_threshold = prob_threshold

    @override
    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategy | None = None,
        adapter: Adapter | None = None,
    ) -> Sequence[HealthcheckAnalysisCard]:
        r"""
        Compute the feasibility of the constraints for the experiment.

        Args:
            experiment: Ax experiment.
            generation_strategy: Ax generation strategy.
            adapter: Ax modelbridge adapter

        Returns:
            A HealthcheckAnalysisCard object with the information on infeasible metrics,
            i.e., metrics for which the constraints are infeasible for all test groups
            (arms).
        """
        status = HealthcheckStatus.PASS
        subtitle = "All constraints are feasible."
        title_status = "Success"
        level = AnalysisCardLevel.LOW
        df = pd.DataFrame()
        category = AnalysisCardCategory.DIAGNOSTIC

        if experiment is None:
            raise UserInputError(
                "ConstraintsFeasibilityAnalysis requires an Experiment."
            )

        if experiment.optimization_config is None:
            subtitle = "No optimization config is specified."
            return [
                self._create_healthcheck_analysis_card(
                    title=f"Ax Constraints Feasibility {title_status}",
                    subtitle=subtitle,
                    df=df,
                    level=level,
                    status=status,
                    category=category,
                ),
            ]

        if (
            experiment.optimization_config.outcome_constraints is None
            or len(experiment.optimization_config.outcome_constraints) == 0
        ):
            subtitle = "No constraints are specified."
            return [
                self._create_healthcheck_analysis_card(
                    title=f"Ax Constraints Feasibility {title_status}",
                    subtitle=subtitle,
                    df=df,
                    level=level,
                    status=status,
                    category=category,
                )
            ]

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        if not is_predictive(adapter=relevant_adapter):
            raise UserInputError(
                "ConstraintsFeasibilityAnalysis requires a predictive model."
            )

        optimization_config = assert_is_instance(
            experiment.optimization_config, OptimizationConfig
        )
        constraints_feasible, df = constraints_feasibility(
            optimization_config=optimization_config,
            model=relevant_adapter,
            prob_threshold=self.prob_threshold,
        )

        if not constraints_feasible:
            status = HealthcheckStatus.WARNING
            subtitle = (
                "The constraints feasibility health check utilizes "
                "samples drawn during the optimization process to assess the "
                "feasibility of constraints set on the experiment. Given these "
                "samples, the model believes there is at least a "
                f"{self.prob_threshold} probability that the constraints will be "
                "violated. We suggest relaxing the bounds for the constraints "
                "on this Experiment."
            )
            title_status = "Warning"

        return [
            self._create_healthcheck_analysis_card(
                title=f"Ax Constraints Feasibility {title_status}",
                subtitle=subtitle,
                df=df,
                level=level,
                status=status,
                category=category,
            ),
        ]


def constraints_feasibility(
    optimization_config: OptimizationConfig,
    model: Adapter,
    prob_threshold: float = 0.99,
) -> tuple[bool, pd.DataFrame]:
    r"""
    Check the feasibility of the constraints for the experiment.

    Args:
        optimization_config: Ax optimization config.
        model: Ax model to use for predictions.
        prob_threshold: Threshold for the probability of constraint violation.

    Returns:
        A tuple of a boolean indicating whether the constraints are feasible and a
        dataframe with information on the probabilities of constraints violation for
        each arm.
    """
    if (optimization_config.outcome_constraints is None) or (
        len(optimization_config.outcome_constraints) == 0
    ):
        raise UserInputError("No constraints are specified.")

    derel_optimization_config = optimization_config
    outcome_constraints = optimization_config.outcome_constraints

    if any(constraint.relative for constraint in outcome_constraints):
        derel_optimization_config = Derelativize().transform_optimization_config(
            optimization_config=optimization_config,
            modelbridge=model,
        )

    constraint_metric_name = [
        constraint.metric.name
        for constraint in derel_optimization_config.outcome_constraints
    ][0]

    arm_dict = get_predictions_by_arm(
        model=model,
        metric_name=constraint_metric_name,
        outcome_constraints=derel_optimization_config.outcome_constraints,
    )

    df = pd.DataFrame(arm_dict)
    constraints_feasible = True
    if all(
        arm_info["overall_probability_constraints_violated"] > prob_threshold
        for arm_info in arm_dict
        if arm_info["arm_name"] != model.status_quo_name
    ):
        constraints_feasible = False

    return constraints_feasible, df
