# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import json

import numpy as np
import pandas as pd

from ax.analysis.analysis import AnalysisCardLevel
from ax.analysis.healthcheck.constraints_feasibility import (
    constraints_feasibility,
    ConstraintsFeasibilityAnalysis,
)
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.exceptions.core import UserInputError
from ax.modelbridge.factory import get_sobol
from ax.modelbridge.generation_node import GenerationNode
from ax.modelbridge.generation_strategy import GenerationStrategy
from ax.modelbridge.model_spec import ModelSpec
from ax.modelbridge.registry import Models
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance, none_throws


class TestConstraintsFeasibilityAnalysis(TestCase):
    @mock_botorch_optimize
    def setUp(self) -> None:
        experiment = get_branin_experiment_with_multi_objective(
            with_batch=False,
            with_status_quo=True,
            with_relative_constraint=True,
        )
        self.df_metric_a = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_a"] * 6,
                "mean": list(np.random.normal(0, 1, 6)),
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
            }
        )
        self.df_metric_b = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_b"] * 6,
                "mean": list(np.random.normal(0, 1, 6)),
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
            }
        )
        self.df_metric_d = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_d"] * 6,
                "mean": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
            }
        )
        df = pd.concat(
            [self.df_metric_a, self.df_metric_b, self.df_metric_d], ignore_index=True
        )

        sobol = get_sobol(search_space=experiment.search_space)
        experiment.new_batch_trial(generator_run=sobol.gen(5))

        batch_trial = assert_is_instance(experiment.trials[0], BatchTrial)

        batch_trial.add_arm(experiment.status_quo)
        batch_trial.set_status_quo_with_weight(
            status_quo=experiment.status_quo, weight=1.0
        )
        experiment.trials[0].mark_running(no_runner_required=True)
        experiment.trials[0].mark_completed()

        experiment.attach_data(data=Data(df=df))

        generation_strategy = GenerationStrategy(
            name="gs",
            nodes=[
                GenerationNode(
                    node_name="gn",
                    model_specs=[
                        ModelSpec(
                            model_enum=Models.BOTORCH_MODULAR,
                        )
                    ],
                )
            ],
        )
        generation_strategy.experiment = experiment
        generation_strategy._fit_current_model(data=experiment.lookup_data())
        self.experiment: Experiment = experiment
        self.generation_strategy: GenerationStrategy = generation_strategy

    @mock_botorch_optimize
    def test_constraints_feasibility(self) -> None:
        self.setUp()
        model = none_throws(self.generation_strategy.model)
        optimization_config = assert_is_instance(
            self.experiment.optimization_config, OptimizationConfig
        )
        constraints_feasible, df_arms = constraints_feasibility(
            optimization_config=optimization_config,
            model=model,
        )
        self.assertTrue(constraints_feasible)

        # with changed data for the constraint metric
        df_metric_d = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_d"] * 6,
                "mean": [0, -1, -2, -3, -4, -5],
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
            }
        )
        df = pd.concat(
            [self.df_metric_a, self.df_metric_b, df_metric_d],
            ignore_index=True,
        )
        experiment = self.experiment
        generation_strategy = self.generation_strategy

        experiment.attach_data(data=Data(df=df))
        generation_strategy._fit_current_model(data=experiment.lookup_data())
        model = none_throws(generation_strategy.model)
        optimization_config = assert_is_instance(
            experiment.optimization_config, OptimizationConfig
        )
        constraints_feasible, df_arms = constraints_feasibility(
            optimization_config=optimization_config, model=model
        )
        self.assertFalse(constraints_feasible)
        experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="branin_a"), minimize=False),
        )
        optimization_config = assert_is_instance(
            experiment.optimization_config, OptimizationConfig
        )
        with self.assertRaises(UserInputError):
            constraints_feasibility(
                optimization_config=optimization_config, model=model
            )

    @mock_botorch_optimize
    def test_compute(self) -> None:
        self.setUp()
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=self.experiment, generation_strategy=self.generation_strategy
        )
        self.assertEqual(card.name, "ConstraintsFeasibility")
        self.assertEqual(card.title, "Ax Constraints Feasibility Success")
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.subtitle, "All constraints are feasible.")

        df_metric_d = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_d"] * 6,
                "mean": [0.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
            }
        )
        df = pd.concat(
            [self.df_metric_a, self.df_metric_b, df_metric_d],
            ignore_index=True,
        )
        experiment = self.experiment
        generation_strategy = self.generation_strategy
        experiment.attach_data(data=Data(df=df))
        generation_strategy.experiment = experiment
        generation_strategy._fit_current_model(data=experiment.lookup_data())
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )
        self.assertEqual(card.name, "ConstraintsFeasibility")
        self.assertEqual(card.title, "Ax Constraints Feasibility Warning")
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        subtitle = (
            "Constraints are infeasible for all test groups (arms) with respect "
            "to the probability threshold 0.95. "
            "We suggest relaxing the constraint bounds for the constraints."
        )
        self.assertEqual(card.subtitle, subtitle)
        self.assertEqual(json.loads(card.blob), {"status": HealthcheckStatus.WARNING})

        # experiment with no constraints
        experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="branin_a"), minimize=False),
            outcome_constraints=[],
        )
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )
        self.assertEqual(card.name, "ConstraintsFeasibility")
        self.assertEqual(card.title, "Ax Constraints Feasibility Success")
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.subtitle, "No constraints are specified.")
        self.assertEqual(json.loads(card.blob), {"status": HealthcheckStatus.PASS})

    def test_no_optimization_config(self) -> None:
        experiment = get_branin_experiment(has_optimization_config=False)
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(experiment=experiment, generation_strategy=None)
        self.assertEqual(card.name, "ConstraintsFeasibility")
        self.assertEqual(card.title, "Ax Constraints Feasibility Success")
        self.assertEqual(card.level, AnalysisCardLevel.LOW)
        self.assertEqual(card.subtitle, "No optimization config is specified.")
        self.assertEqual(json.loads(card.blob), {"status": HealthcheckStatus.PASS})
