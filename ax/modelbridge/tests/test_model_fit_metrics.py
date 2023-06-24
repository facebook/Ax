#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import cast, Dict

from ax.core.experiment import Experiment
from ax.core.objective import Objective
from ax.core.optimization_config import OptimizationConfig
from ax.metrics.branin import BraninMetric
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.runners.synthetic import SyntheticRunner
from ax.service.scheduler import get_fitted_model_bridge, Scheduler, SchedulerOptions
from ax.utils.common.constants import Keys
from ax.utils.common.testutils import TestCase
from ax.utils.testing.core_stubs import get_branin_search_space

NUM_SOBOL = 5


class TestModelBridgeFitMetrics(TestCase):
    def setUp(self) -> None:
        # setting up experiment and generation strategy
        self.runner = SyntheticRunner()
        self.branin_experiment = Experiment(
            name="branin_test_experiment",
            search_space=get_branin_search_space(),
            runner=self.runner,
            optimization_config=OptimizationConfig(
                objective=Objective(
                    metric=BraninMetric(name="branin", param_names=["x1", "x2"]),
                    minimize=True,
                ),
            ),
            is_test=True,
        )
        self.branin_experiment._properties[
            Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF
        ] = True
        self.generation_strategy = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL, num_trials=NUM_SOBOL, max_parallelism=NUM_SOBOL
                ),
                GenerationStep(model=Models.GPEI, num_trials=-1),
            ]
        )

    def test_model_fit_metrics(self) -> None:
        scheduler = Scheduler(
            experiment=self.branin_experiment,
            generation_strategy=self.generation_strategy,
            options=SchedulerOptions(),
        )
        # need to run some trials to initialize the ModelBridge
        scheduler.run_n_trials(max_trials=NUM_SOBOL + 1)
        model_bridge = get_fitted_model_bridge(scheduler)

        # testing ModelBridge.compute_model_fit_metrics with default metrics
        fit_metrics = model_bridge.compute_model_fit_metrics(self.branin_experiment)
        r2 = fit_metrics.get("coefficient_of_determination")
        self.assertIsInstance(r2, dict)
        r2 = cast(Dict[str, float], r2)
        self.assertTrue("branin" in r2)
        r2_branin = r2["branin"]
        self.assertIsInstance(r2_branin, float)

        std = fit_metrics.get("std_of_the_standardized_error")
        self.assertIsInstance(std, dict)
        std = cast(Dict[str, float], std)
        self.assertTrue("branin" in std)
        std_branin = std["branin"]
        self.assertIsInstance(std_branin, float)

        # testing with empty metrics
        empty_metrics = model_bridge.compute_model_fit_metrics(
            self.branin_experiment, fit_metrics_dict={}
        )
        self.assertIsInstance(empty_metrics, dict)
        self.assertTrue(len(empty_metrics) == 0)
