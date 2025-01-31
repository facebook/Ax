# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json
from typing import Tuple

import pandas as pd

from ax.analysis.analysis import AnalysisCardLevel

from ax.analysis.healthcheck.healthcheck_analysis import (
    HealthcheckAnalysis,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.plotly.arm_effects.utils import get_predictions_by_arm
from ax.analysis.plotly.utils import is_predictive
from ax.core.experiment import Experiment
from ax.core.generation_strategy_interface import GenerationStrategyInterface
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.transforms.derelativize import Derelativize
from pyre_extensions import assert_is_instance, none_throws


class ConstraintsFeasibilityAnalysis(HealthcheckAnalysis):
    """
    Analysis for checking the feasibility of the constraints for the experiment.
    A constraint is considered feasible if the probability of constraints violation
    is below the threshold for at least one arm.
    """

    def __init__(self, prob_threshold: float = 0.95) -> None:
        r"""
        Args:
            prob_theshold: The threshold for the probability of constraint violation.

        Returns None
        """
        self.prob_threshold = prob_threshold

    def compute(
        self,
        experiment: Experiment | None = None,
        generation_strategy: GenerationStrategyInterface | None = None,
    ) -> HealthcheckAnalysisCard:
        r"""
        Compute the feasibility of the constraints for the experiment.

        Args:
            experiment: Ax experiment.
            generation_strategy: Ax generation strategy.
            prob_threhshold: Threshold for the probability of constraint violation.
                Constraints are considered feasible if the probability of constraint
                violation is below the threshold for at least one arm.

        Returns:
            A HealthcheckAnalysisCard object with the information on infeasible metrics,
            i.e., metrics for which the constraints are infeasible for all test groups
            (arms).
        """
        status = HealthcheckStatus.PASS
        subtitle = "All constraints are feasible."
        title_status = "Success"
        level = AnalysisCardLevel.LOW
        df = pd.DataFrame({"status": [status]})

        if experiment is None:
            raise UserInputError(
                "ConstraintsFeasibilityAnalysis requires an Experiment."
            )

        if experiment.optimization_config is None:
            subtitle = "No optimization config is specified."
            return HealthcheckAnalysisCard(
                name="ConstraintsFeasibility",
                title=f"Ax Constraints Feasibility {title_status}",
                blob=json.dumps({"status": status}),
                subtitle=subtitle,
                df=df,
                level=level,
            )

        if (
            experiment.optimization_config.outcome_constraints is None
            or len(experiment.optimization_config.outcome_constraints) == 0
        ):
            subtitle = "No constraints are specified."
            return HealthcheckAnalysisCard(
                name="ConstraintsFeasibility",
                title=f"Ax Constraints Feasibility {title_status}",
                blob=json.dumps({"status": status}),
                subtitle=subtitle,
                df=df,
                level=level,
            )

        if generation_strategy is None:
            raise UserInputError(
                "ConstraintsFeasibilityAnalysis requires a GenerationStrategy."
            )
        generation_strategy = assert_is_instance(
            generation_strategy, GenerationStrategy
        )

        if generation_strategy.model is None:
            generation_strategy._fit_current_model(data=experiment.lookup_data())

        model = none_throws(generation_strategy.model)
        if not is_predictive(model=model):
            raise UserInputError(
                "ConstraintsFeasibility requires a GenerationStrategy which is "
                "in a state where the current model supports prediction. "
                f"The current model is {model._model_key} and does not support "
                "prediction."
            )
        optimization_config = assert_is_instance(
            experiment.optimization_config, OptimizationConfig
        )
        constraints_feasible, df = constraints_feasibility(
            optimization_config=optimization_config,
            model=model,
            prob_threshold=self.prob_threshold,
        )
        df["status"] = status

        if not constraints_feasible:
            status = HealthcheckStatus.WARNING
            subtitle = (
                "Constraints are infeasible for all test groups (arms) with respect "
                f"to the probability threshold {self.prob_threshold}. "
                "We suggest relaxing the constraint bounds for the constraints."
            )
            title_status = "Warning"
            df.loc[
                df["overall_probability_constraints_violated"] > self.prob_threshold,
                "status",
            ] = status

        return HealthcheckAnalysisCard(
            name="ConstraintsFeasibility",
            title=f"Ax Constraints Feasibility {title_status}",
            blob=json.dumps({"status": status}),
            subtitle=subtitle,
            df=df,
            level=level,
        )


def constraints_feasibility(
    optimization_config: OptimizationConfig,
    model: ModelBridge,
    prob_threshold: float = 0.99,
) -> Tuple[bool, pd.DataFrame]:
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
