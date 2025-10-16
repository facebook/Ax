# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from typing import final

import pandas as pd
from ax.adapter.base import Adapter
from ax.analysis.analysis import Analysis
from ax.analysis.healthcheck.healthcheck_analysis import (
    create_healthcheck_analysis_card,
    HealthcheckAnalysisCard,
    HealthcheckStatus,
)
from ax.analysis.utils import (
    extract_relevant_adapter,
    POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD,
    prepare_arm_data,
)
from ax.core.experiment import Experiment
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.generation_strategy.generation_strategy import GenerationStrategy
from pyre_extensions import assert_is_instance, override


@final
class ConstraintsFeasibilityAnalysis(Analysis):
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
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
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
            return create_healthcheck_analysis_card(
                name=self.__class__.__name__,
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

        arm_data = prepare_arm_data(
            experiment=experiment,
            metric_names=[*optimization_config.metrics.keys()],
            use_model_predictions=True,
            adapter=relevant_adapter,
        )

        constraints_feasible = (
            arm_data["p_feasible_mean"] > POSSIBLE_CONSTRAINT_VIOLATION_THRESHOLD
        ).all()

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

        return create_healthcheck_analysis_card(
            name=self.__class__.__name__,
            title=f"Ax Constraints Feasibility {title_status}",
            subtitle=subtitle,
            df=df,
            status=status,
        )
