# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import numpy as np
import pandas as pd
from ax.adapter.factory import get_sobol
from ax.adapter.registry import Generators

from ax.analysis.healthcheck.constraints_feasibility import (
    ConstraintsFeasibilityAnalysis,
)
from ax.analysis.healthcheck.healthcheck_analysis import HealthcheckStatus
from ax.core.batch_trial import BatchTrial
from ax.core.data import Data
from ax.core.experiment import Experiment
from ax.core.metric import Metric
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generation_strategy import GenerationStrategy
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import (
    get_branin_experiment,
    get_branin_experiment_with_multi_objective,
)
from ax.utils.testing.mock import mock_botorch_optimize
from pyre_extensions import assert_is_instance


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
                "metric_signature": ["branin_a"] * 6,
            }
        )
        self.df_metric_b = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_b"] * 6,
                "mean": list(np.random.normal(0, 1, 6)),
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
                "metric_signature": ["branin_b"] * 6,
            }
        )
        self.df_metric_d = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_d"] * 6,
                "mean": [1.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
                "metric_signature": ["branin_d"] * 6,
            }
        )
        df = pd.concat(
            [self.df_metric_a, self.df_metric_b, self.df_metric_d], ignore_index=True
        )

        sobol = get_sobol(search_space=experiment.search_space)
        experiment.new_batch_trial(generator_run=sobol.gen(5))

        batch_trial = assert_is_instance(experiment.trials[0], BatchTrial)

        batch_trial.add_arm(experiment.status_quo)
        batch_trial.add_status_quo_arm(weight=1.0)
        experiment.trials[0].mark_running(no_runner_required=True)
        experiment.trials[0].mark_completed()

        experiment.attach_data(data=Data(df=df))

        generation_strategy = GenerationStrategy(
            name="gs",
            nodes=[
                GenerationNode(
                    name="gn",
                    generator_specs=[
                        GeneratorSpec(
                            generator_enum=Generators.BOTORCH_MODULAR,
                        )
                    ],
                )
            ],
        )
        generation_strategy.experiment = experiment
        generation_strategy._curr._fit(experiment=experiment)
        self.experiment: Experiment = experiment
        self.generation_strategy: GenerationStrategy = generation_strategy

    @mock_botorch_optimize
    def test_compute(self) -> None:
        self.setUp()
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=self.experiment, generation_strategy=self.generation_strategy
        )
        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Constraints Feasibility Success")
        self.assertEqual(card.subtitle, "All constraints are feasible.")

        df_metric_d = pd.DataFrame(
            {
                "arm_name": ["status_quo", "0_0", "0_1", "0_2", "0_3", "0_4"],
                "metric_name": ["branin_d"] * 6,
                "mean": [1.0, -1.0, -2.0, -3.0, -4.0, -5.0],
                "sem": [0.1] * 6,
                "trial_index": [0] * 6,
                "metric_signature": ["branin_d"] * 6,
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
        generation_strategy._curr._fit(experiment=experiment)
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )
        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Constraints Feasibility Warning")
        subtitle = (
            "The constraints feasibility health check utilizes "
            "samples drawn during the optimization process to assess the "
            "feasibility of constraints set on the experiment. Given these "
            "samples, the model believes there is at least a "
            "0.95 probability that the constraints will be "
            "violated. We suggest relaxing the bounds for the constraints "
            "on this Experiment."
        )
        self.assertEqual(card.subtitle, subtitle)
        self.assertEqual(card.get_status(), HealthcheckStatus.WARNING)

        # experiment with no constraints
        experiment.optimization_config = OptimizationConfig(
            objective=Objective(metric=Metric(name="branin_a"), minimize=False),
            outcome_constraints=[],
        )
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(
            experiment=experiment, generation_strategy=generation_strategy
        )
        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Constraints Feasibility Success")
        self.assertEqual(card.subtitle, "No constraints are specified.")
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)

    def test_no_optimization_config(self) -> None:
        experiment = get_branin_experiment(has_optimization_config=False)
        cfa = ConstraintsFeasibilityAnalysis()
        card = cfa.compute(experiment=experiment, generation_strategy=None)
        self.assertEqual(card.name, "ConstraintsFeasibilityAnalysis")
        self.assertEqual(card.title, "Ax Constraints Feasibility Success")
        self.assertEqual(card.subtitle, "No optimization config is specified.")
        self.assertEqual(card.get_status(), HealthcheckStatus.PASS)
