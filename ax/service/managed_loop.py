#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable

from ax.core.arm import Arm
from ax.core.base_trial import BaseTrial
from ax.core.batch_trial import BatchTrial
from ax.core.experiment import Experiment
from ax.core.formatting_utils import data_and_evaluations_from_raw_data
from ax.core.trial import Trial
from ax.core.types import (
    TEvaluationFunction,
    TEvaluationOutcome,
    TModelPredictArm,
    TParameterization,
)
from ax.core.utils import get_pending_observation_features
from ax.exceptions.constants import CHOLESKY_ERROR_ANNOTATION
from ax.exceptions.core import SearchSpaceExhausted, UserInputError
from ax.modelbridge.base import ModelBridge
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.registry import Models
from ax.service.utils.best_point import (
    get_best_parameters_from_model_predictions_with_trial_index,
    get_best_raw_objective_point,
)
from ax.service.utils.instantiation import (
    DEFAULT_OBJECTIVE_NAME,
    InstantiationBase,
    TParameterRepresentation,
)
from ax.utils.common.executils import retry_on_exception
from ax.utils.common.logger import get_logger
from pyre_extensions import none_throws


logger: logging.Logger = get_logger(__name__)


class OptimizationLoop:
    """Managed optimization loop, in which Ax oversees deployment of trials and
    gathering data."""

    def __init__(
        self,
        experiment: Experiment,
        evaluation_function: TEvaluationFunction,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        random_seed: int | None = None,
        wait_time: int = 0,
        run_async: bool = False,  # TODO[Lena],
        generation_strategy: GenerationStrategy | None = None,
    ) -> None:
        assert not run_async, "OptimizationLoop does not yet support async."
        self.wait_time = wait_time
        self.total_trials = total_trials
        self.arms_per_trial = arms_per_trial
        self.random_seed = random_seed
        self.evaluation_function: TEvaluationFunction = evaluation_function
        assert len(experiment.trials) == 0, (
            "Optimization Loop should not be initialized with an experiment "
            "that has trials already."
        )
        self.experiment = experiment
        if generation_strategy is None:
            # pyre-fixme[4]: Attribute must be annotated.
            self.generation_strategy = choose_generation_strategy(
                search_space=experiment.search_space,
                use_batch_trials=self.arms_per_trial > 1,
                random_seed=self.random_seed,
                experiment=experiment,
            )
        else:
            self.generation_strategy = generation_strategy
        self.current_trial = 0

    @staticmethod
    def with_evaluation_function(
        parameters: list[TParameterRepresentation],
        evaluation_function: TEvaluationFunction,
        experiment_name: str | None = None,
        objective_name: str | None = None,
        minimize: bool = False,
        parameter_constraints: list[str] | None = None,
        outcome_constraints: list[str] | None = None,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        wait_time: int = 0,
        random_seed: int | None = None,
        generation_strategy: GenerationStrategy | None = None,
    ) -> OptimizationLoop:
        """Constructs a synchronous `OptimizationLoop` using an evaluation
        function."""
        if objective_name is None:
            objective_name = DEFAULT_OBJECTIVE_NAME
        experiment = InstantiationBase.make_experiment(
            name=experiment_name,
            parameters=parameters,
            objectives={objective_name: "minimize" if minimize else "maximize"},
            parameter_constraints=parameter_constraints,
            outcome_constraints=outcome_constraints,
        )
        return OptimizationLoop(
            experiment=experiment,
            total_trials=total_trials,
            arms_per_trial=arms_per_trial,
            random_seed=random_seed,
            wait_time=wait_time,
            generation_strategy=generation_strategy,
            evaluation_function=evaluation_function,
        )

    @classmethod
    def with_runners_and_metrics(
        cls,
        parameters: list[TParameterRepresentation],
        path_to_runner: str,
        paths_to_metrics: list[str],
        experiment_name: str | None = None,
        objective_name: str | None = None,
        minimize: bool = False,
        parameter_constraints: list[str] | None = None,
        outcome_constraints: list[str] | None = None,
        total_trials: int = 20,
        arms_per_trial: int = 1,
        wait_time: int = 0,
        random_seed: int | None = None,
    ) -> OptimizationLoop:
        """Constructs an asynchronous `OptimizationLoop` using Ax runners and
        metrics."""
        # NOTE: Could use `Scheduler` to implement this if needed.
        raise NotImplementedError

    def _call_evaluation_function(
        self, parameterization: TParameterization, weight: float | None = None
    ) -> TEvaluationOutcome:
        signature = inspect.signature(self.evaluation_function)
        num_evaluation_function_params = len(signature.parameters.items())
        if num_evaluation_function_params == 1:
            # pyre-ignore [20]: Can't run instance checks on subscripted generics.
            evaluation = self.evaluation_function(parameterization)
        elif num_evaluation_function_params == 2:
            # pyre-ignore [19]: Can't run instance checks on subscripted generics.
            evaluation = self.evaluation_function(parameterization, weight)
        else:
            raise UserInputError(
                "Evaluation function must take either one parameter "
                "(parameterization) or two parameters (parameterization and weight)."
            )

        return evaluation

    def _get_new_trial(self) -> BaseTrial:
        if self.arms_per_trial == 1:
            return self.experiment.new_trial(
                generator_run=self.generation_strategy.gen(
                    experiment=self.experiment,
                    pending_observations=get_pending_observation_features(
                        experiment=self.experiment
                    ),
                )
            )
        elif self.arms_per_trial > 1:
            return self.experiment.new_batch_trial(
                generator_run=self.generation_strategy.gen(
                    experiment=self.experiment, n=self.arms_per_trial
                )
            )
        else:
            raise UserInputError(
                f"Invalid number of arms per trial: {self.arms_per_trial}"
            )

    def _get_weights_by_arm(
        self, trial: BaseTrial
    ) -> Iterable[tuple[Arm, float | None]]:
        if isinstance(trial, Trial):
            if trial.arm is not None:
                return [(none_throws(trial.arm), None)]
            return []
        elif isinstance(trial, BatchTrial):
            return trial.normalized_arm_weights().items()
        else:
            raise UserInputError(f"Invalid trial type: {type(trial)}")

    @retry_on_exception(
        logger=logger,
        exception_types=(RuntimeError,),
        suppress_all_errors=False,
        wrap_error_message_in=CHOLESKY_ERROR_ANNOTATION,
    )
    def run_trial(self) -> None:
        """Run a single step of the optimization plan."""
        if self.current_trial >= self.total_trials:
            raise ValueError("Optimization is complete, cannot run another trial.")
        logger.info(f"Running optimization trial {self.current_trial + 1}...")

        trial = self._get_new_trial()

        trial.mark_running(no_runner_required=True)
        _, data = data_and_evaluations_from_raw_data(
            raw_data={
                arm.name: self._call_evaluation_function(arm.parameters, weight)
                for arm, weight in self._get_weights_by_arm(trial)
            },
            trial_index=self.current_trial,
            sample_sizes={},
            data_type=self.experiment.default_data_type,
            metric_names=none_throws(
                self.experiment.optimization_config
            ).objective.metric_names,
        )

        self.experiment.attach_data(data=data)
        trial.mark_completed()
        self.current_trial += 1

    def full_run(self) -> OptimizationLoop:
        """Runs full optimization loop as defined in the provided optimization
        plan."""
        num_steps = self.total_trials
        logger.info(f"Started full optimization with {num_steps} steps.")
        for _ in range(num_steps):
            try:
                self.run_trial()
            except SearchSpaceExhausted as err:
                logger.info(
                    f"Stopped optimization as the search space is exhaused. Message "
                    f"from generation strategy: {err}."
                )
                return self
            except Exception:
                logger.exception("Encountered exception during optimization: ")
                return self
        return self

    def get_best_point(self) -> tuple[TParameterization, TModelPredictArm | None]:
        """Obtains the best point encountered in the course
        of this optimization."""
        # Find latest trial which has a generator_run attached and get its predictions
        best_point = get_best_parameters_from_model_predictions_with_trial_index(
            experiment=self.experiment, models_enum=Models
        )
        if best_point is not None:
            _, parameterizations, predictions = best_point
            return parameterizations, predictions

        # Could not find through model, default to using raw objective.
        parameterization, values = get_best_raw_objective_point(
            experiment=self.experiment
        )
        # For values, grab just the means to conform to TModelPredictArm format.
        return (
            parameterization,
            (
                {k: v[0] for k, v in values.items()},  # v[0] is mean
                {k: {k: v[1] * v[1]} for k, v in values.items()},  # v[1] is sem
            ),
        )

    def get_current_model(self) -> ModelBridge | None:
        """Obtain the most recently used model in optimization."""
        return self.generation_strategy.model


def optimize(
    parameters: list[TParameterRepresentation],
    evaluation_function: TEvaluationFunction,
    experiment_name: str | None = None,
    objective_name: str | None = None,
    minimize: bool = False,
    parameter_constraints: list[str] | None = None,
    outcome_constraints: list[str] | None = None,
    total_trials: int = 20,
    arms_per_trial: int = 1,
    random_seed: int | None = None,
    generation_strategy: GenerationStrategy | None = None,
) -> tuple[TParameterization, TModelPredictArm | None, Experiment, ModelBridge | None]:
    """Construct and run a full optimization loop."""
    loop = OptimizationLoop.with_evaluation_function(
        parameters=parameters,
        objective_name=objective_name,
        evaluation_function=evaluation_function,
        experiment_name=experiment_name,
        minimize=minimize,
        parameter_constraints=parameter_constraints,
        outcome_constraints=outcome_constraints,
        total_trials=total_trials,
        arms_per_trial=arms_per_trial,
        random_seed=random_seed,
        generation_strategy=generation_strategy,
    )
    loop.full_run()
    parameterization, values = loop.get_best_point()
    return parameterization, values, loop.experiment, loop.get_current_model()
