# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import pandas as pd
from ax.adapter.base import Adapter
from ax.adapter.transforms.derelativize import Derelativize

from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.plotly.utils import get_predictions_by_arm
from ax.analysis.utils import extract_relevant_adapter
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
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
    ) -> HealthcheckAnalysisCard:
        r"""
        Compute the feasibility of the constraints for the experiment.

        Args:
            experiment: Ax experiment.
            generation_strategy: Ax generation strategy.
            adapter: Ax adapter adapter

        Returns:
            A HealthcheckAnalysisCard object with the information on infeasible metrics,
            i.e., metrics for which the constraints are infeasible for all test groups
            (arms).
        """
        status = HealthcheckStatus.PASS
        subtitle = "All constraints are feasible."
        title_status = "Success"
        df = pd.DataFrame()

        if experiment is None:
            raise UserInputError(
                "ConstraintsFeasibilityAnalysis requires an Experiment."
            )

        if experiment.optimization_config is None:
            subtitle = "No optimization config is specified."
            return self._create_healthcheck_analysis_card(
                title=f"Ax Constraints Feasibility {title_status}",
                subtitle=subtitle,
                df=df,
                status=status,
            )

        if (
            experiment.optimization_config.outcome_constraints is None
            or len(experiment.optimization_config.outcome_constraints) == 0
        ):
            subtitle = "No constraints are specified."
            return self._create_healthcheck_analysis_card(
                title=f"Ax Constraints Feasibility {title_status}",
                subtitle=subtitle,
                df=df,
                status=status,
            )

        relevant_adapter = extract_relevant_adapter(
            experiment=experiment,
            generation_strategy=generation_strategy,
            adapter=adapter,
        )

        if (
            not relevant_adapter.can_predict
        ):  # TODO: Verify that we actually need to predict OOS here
            raise UserInputError(
                "ConstraintsFeasibilityAnalysis requires an adapter that can "
                "make predictions for unobserved outcomes"
            )

        optimization_config = assert_is_instance(
            experiment.optimization_config, OptimizationConfig
        )
        constraints_feasible, df = constraints_feasibility(
            optimization_config=optimization_config,
            adapter=relevant_adapter,
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

        return self._create_healthcheck_analysis_card(
            title=f"Ax Constraints Feasibility {title_status}",
            subtitle=subtitle,
            df=df,
            status=status,
        )


def constraints_feasibility(
    optimization_config: OptimizationConfig,
    adapter: Adapter,
    prob_threshold: float = 0.99,
) -> tuple[bool, pd.DataFrame]:
    r"""
    Check the feasibility of the constraints for the experiment.

    Args:
        optimization_config: Ax optimization config.
        adapter: Ax adapter to use for predictions.
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
            adapter=adapter,
        )

    constraint_metric_name = [
        constraint.metric.name
        for constraint in derel_optimization_config.outcome_constraints
    ][0]

    arm_dict = get_predictions_by_arm(
        adapter=adapter,
        metric_name=constraint_metric_name,
        outcome_constraints=derel_optimization_config.outcome_constraints,
    )

    df = pd.DataFrame(arm_dict)
    constraints_feasible = True
    if all(
        arm_info["overall_probability_constraints_violated"] > prob_threshold
        for arm_info in arm_dict
        if arm_info["arm_name"] != adapter.status_quo_name
    ):
        constraints_feasible = False

    return constraints_feasible, df
